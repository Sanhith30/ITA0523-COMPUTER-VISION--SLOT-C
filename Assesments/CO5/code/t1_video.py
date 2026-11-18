"""TASK 1 - Input stage: compare three ways of reading a live 1080p/25 fps stream.
A live camera is emulated with a *virtual real-time clock*: frame i is captured at t0 + i/25 s. Frames are really
decoded from a 1080p MJPG file with cv2.VideoCapture when a consumer asks for them, and a bounded jitter buffer
(socket/decoder buffer, default 25 frames = 1 s) drops the oldest frame on overflow - the same behaviour as an RTSP source."""
import time, threading, queue, sys, platform
import numpy as np, cv2, psutil
from common import *

BUDGET_MS, FPS = 40.0, 25
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def work(frame, extra_ms=0):
    """Normal processing = Haar face detection on a 480x270 grey copy (~23 ms on this single-core machine);
    extra_ms = the artificial processing delay required by the assignment (80 ms)."""
    g = cv2.equalizeHist(cv2.cvtColor(cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    cascade.detectMultiScale(g, 1.25, 5, minSize=(36, 36))
    if extra_ms: time.sleep(extra_ms / 1000.0)

class LiveStream:
    def __init__(self, path, fps=FPS, buf=25):
        self.cap = cv2.VideoCapture(path); self.N = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps, self.buf = fps, buf
        self.t0 = time.perf_counter() + 0.05          # capture time of frame 0
        self.i = 0; self.pos = 0; self.overflow = 0; self.last_idx = -1
    def newest(self): return int((time.perf_counter() - self.t0) * self.fps)
    def t_cap(self, idx): return self.t0 + idx / self.fps
    def _next_index(self):
        n = self.newest()
        if n - self.i >= self.buf:                    # buffer overflow: oldest frames are lost
            new_i = n - self.buf + 1; self.overflow += new_i - self.i; self.i = new_i
        wait = self.t_cap(self.i) - time.perf_counter()
        if wait > 0: time.sleep(wait)                 # frame not captured yet: block like cap.read()
        idx = self.i; self.i += 1; return idx
    def _seek(self, idx):
        fi = idx % self.N
        if fi != self.pos: self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        self.pos = fi + 1
    def read(self):                                   # = grab() + retrieve()
        idx = self._next_index(); self._seek(idx); ok, f = self.cap.read(); return f, idx, self.t_cap(idx)
    def grab(self):
        idx = self._next_index(); self._seek(idx); self.cap.grab(); self.last_idx = idx; return idx
    def retrieve(self):
        ok, f = self.cap.retrieve(); return f

def run(approach, extra_ms, duration=12.0, buf=25, k=2, qmax=30):
    s = LiveStream(os.path.join(VID, "gate_1080p.avi"), buf=buf)
    proc = psutil.Process(); c0 = proc.cpu_times(); w0 = time.perf_counter()
    ages, ends, series, proc_ms = [], [], [], []
    stop = threading.Event(); st = dict(lat=None, seq=0, used=0, over=0, qdrop=0)
    cond = threading.Condition(); q = queue.Queue(maxsize=qmax)
    def reader_latest():
        while not stop.is_set():
            f, idx, tc = s.read()
            with cond:
                if st["lat"] is not None and st["seq"] != st["used"]: st["over"] += 1   # unused frame replaced
                st["lat"] = (f, idx, tc); st["seq"] += 1; cond.notify()
    def reader_queue():
        while not stop.is_set():
            item = s.read()
            try: q.put_nowait(item)
            except queue.Full:
                try: q.get_nowait(); st["qdrop"] += 1
                except queue.Empty: pass
                q.put_nowait(item)
    if approach == "B": th = threading.Thread(target=reader_latest, daemon=True); th.start()
    if approach == "B2": th = threading.Thread(target=reader_queue, daemon=True); th.start()
    last_seq = 0; processed = 0
    while time.perf_counter() - w0 < duration:
        if approach == "A":
            f, idx, tc = s.read()
        elif approach == "C":
            idx = s.grab()
            if idx % k: continue
            f = s.retrieve(); tc = s.t_cap(idx)
        elif approach == "B":
            with cond:
                while st["seq"] == last_seq and not stop.is_set():
                    cond.wait(0.05)
                    if time.perf_counter() - w0 >= duration: break
                if st["seq"] == last_seq: break
                f, idx, tc = st["lat"]; last_seq = st["seq"]; st["used"] = last_seq
        else:
            try: f, idx, tc = q.get(timeout=0.2)
            except queue.Empty: continue
        t_start = time.perf_counter(); age = (t_start - tc) * 1000
        work(f, extra_ms); t_end = time.perf_counter()
        ages.append(age); ends.append((t_end - tc) * 1000); proc_ms.append((t_end - t_start) * 1000)
        series.append((t_end - w0, age)); processed += 1
    stop.set()
    wall = time.perf_counter() - w0; c1 = proc.cpu_times()
    cpu = ((c1.user - c0.user) + (c1.system - c0.system)) / wall * 100
    captured = int(wall * FPS)
    a = np.array(ages)
    return dict(approach=approach, extra_ms=extra_ms, duration=wall, processed=processed, captured=captured,
                fps=processed / wall, age_mean=float(a.mean()), age_p95=float(np.percentile(a, 95)), age_max=float(a.max()),
                result_latency_mean=float(np.mean(ends)), proc_ms_mean=float(np.mean(proc_ms)),
                not_processed=captured - processed, overflow_drops=s.overflow + st["over"] + st["qdrop"],
                cpu_pct=cpu, frames_within_budget=float(np.mean(np.array(proc_ms) <= BUDGET_MS) * 100),
                series=series[::3])

if __name__ == "__main__":
    out = {"runs": []}
    for extra in (0, 80):
        for ap in ("A", "B", "B2", "C"):
            r = run(ap, extra); out["runs"].append(r)
            print(ap, extra, f"fps={r['fps']:.1f} age={r['age_mean']:.0f}ms p95={r['age_p95']:.0f} notproc={r['not_processed']} ovf={r['overflow_drops']} cpu={r['cpu_pct']:.0f}%", flush=True)
    r = run("A", 80, duration=20.0, buf=10**6); r["label"] = "A, unbounded buffer, 80 ms"; out["unbounded"] = r
    print("unbounded", f"age_max={r['age_max']:.0f} ms at 20 s", flush=True)
    # decode cost micro-benchmark (software FFmpeg decode of 1080p MJPG)
    cap = cv2.VideoCapture(os.path.join(VID, "gate_1080p.avi"))
    out["decode_ms"] = dict(read=timeit(lambda: cap.read(), 30)[0], grab=timeit(lambda: cap.grab(), 30)[0])
    cap.set(1, 0); cap.grab(); out["decode_ms"]["retrieve"] = timeit(lambda: cap.retrieve(), 30)[0]
    # hardware acceleration request (needs GPU/VAAPI/D3D11 - not available here)
    try:
        c2 = cv2.VideoCapture(os.path.join(VID, "gate_1080p.avi"), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        out["hw_accel_prop"] = c2.get(cv2.CAP_PROP_HW_ACCELERATION); out["hw_accel_opened"] = c2.isOpened()
    except Exception as e: out["hw_accel_prop"] = str(e)
    bi = cv2.getBuildInformation(); out["build"] = {l.strip() for l in bi.splitlines() if ("FFMPEG" in l or "GStreamer" in l)} and [l.strip() for l in bi.splitlines() if ("FFMPEG:" in l or "GStreamer:" in l)]
    # batching micro-benchmark with the ArcFace-MobileFaceNet ONNX model (112x112 crops)
    try:
        arc = cv2.dnn.readNetFromONNX(os.path.join(MODELS, "buffalo_sc", "w600k_mbf.onnx"))
        res = {}
        for B in (1, 2, 4, 8, 16):
            imgs = [np.random.randint(0, 255, (112, 112, 3), np.uint8) for _ in range(B)]
            def fn():
                blob = cv2.dnn.blobFromImages(imgs, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True); arc.setInput(blob); return arc.forward()
            t = timeit(fn, 5, 1)[0]; res[B] = dict(batch_ms=t, per_image_ms=t / B, out_shape=list(fn().shape))
        out["batching"] = res
    except Exception as e: out["batching"] = str(e)
    out["env"] = dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__, cores=psutil.cpu_count())
    save_json("t1.json", out); print("done")
