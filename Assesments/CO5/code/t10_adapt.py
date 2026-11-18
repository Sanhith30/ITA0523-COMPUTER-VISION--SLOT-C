"""TASK 10 - Night parking lot: 12 streams x 15 fps (1080p cameras; processed here at 640x360 on one core). Strategies: A frame skipping + ROI | B motion-triggered cascade | C lighter IR-suited techniques."""
import cv2, numpy as np, os, time, collections, psutil, re
from common import *
import t8_video_recog as V8
import t6_ocr as P6
import t3_detect as T3
import t3_data as D
V8.EXP = 1.0; FPS, NR, NS = 15, 90, 12; W, H = 640, 360
rng = np.random.default_rng(100)
HOG, _ = T3.train_hog(); 
def person(f, t, subj=3):                                          # person = body ellipse + ORL face, entering at t=30
    if t < 30: return None
    x = int(100 + (t - 30) * 5); cv2.ellipse(f, (x + 38, 255), (45, 90), 0, 0, 360, (35, 35, 40), -1); D.paste(f, D.orl(subj, 7), x, 120, 76, 92); return (x, 120, x + 76, 212)
def base(kind):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); lamp = np.exp(-(((xx - 480) ** 2 + (yy - 60) ** 2) / (2 * 140 ** 2)))
    b = np.clip(18 + 20 * yy / H + 70 * lamp, 0, 255); f = np.stack([b, b, b * 1.05], -1).astype(np.uint8); cv2.rectangle(f, (0, 280), (W, 360), (22, 22, 24), -1); return f
def clip(kind, seed):
    r = np.random.default_rng(seed); fr = []; meta = dict(kind=kind, event=None); plate = P6.rand_plate() if kind == "IR vehicle" else None
    if plate: ps = cv2.resize(P6.make_scene(plate, "dirt"), (W, H))
    for t in range(NR):
        f = base(kind)
        if kind in ("night person", "glare person"):
            bb = person(f, t); meta["event"] = 30
        if kind == "IR vehicle" and t >= 30: f = (0.55 * ps + 0.45 * f).astype(np.uint8) if False else ps.copy(); meta["event"] = 30
        if kind == "tree (empty)":
            for k in range(8): cv2.circle(f, (int(300 + 90 * np.sin(0.4 * t + k)), 40 + 35 * k), 14, (12, 12, 12), -1)
            f = np.clip(f * (1 + 0.5 * (t % 30 > 20)), 0, 255).astype(np.uint8)
        if kind == "glare person":
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); veil = 0.7 * np.exp(-(((xx - 330) ** 2 + (yy - 170) ** 2) / (2 * 110 ** 2))); f = np.clip(f + 255 * veil[..., None] * (t >= 40), 0, 255).astype(np.uint8)
        if kind in ("IR vehicle",) or kind == "IR person":
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); f = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        f = np.clip(f.astype(np.float32) + r.normal(0, 4, f.shape), 0, 255).astype(np.uint8); fr.append(f)
    meta["plate"] = plate; return fr, meta
KINDS = ["night person", "IR vehicle", "glare person", "tree (empty)"]; STREAMS = [clip(KINDS[i % 4], i) for i in range(NS)]
FACE_LB = None
def heavy_orig(f):                                                  # original chain: SCRFD + ArcFace + plate localise/deep OCR (ROI = lower 70 %)
    roi = f[100:, :]; faces = V8.SYS.detect(roi); nf = len(faces)
    for b in faces: V8.emb(roi, b, 1.0)
    cn = P6.localise(f); txt = P6.ocr_C(P6.rectify(f, cn))[0] if cn is not None else None
    return nf > 0, txt
def light_chain(f):                                                 # CLAHE + HOG face/people-like detector at 320x180 + classical plate OCR
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)); r, w = HOG.detectMultiScale(g[100:], 0.5, (8, 8), (0, 0), 1.15, 2)
    cn = P6.localise(f); txt = P6.ocr_B(P6.rectify(f, cn))[0] if cn is not None else None
    return len(r) > 0, txt
class MG:
    def __init__(s): s.m = cv2.createBackgroundSubtractorMOG2(120, 30, False)
    def active(s, f):
        m = s.m.apply(cv2.resize(f[100:], (160, 65), interpolation=cv2.INTER_AREA)); return (m > 200).mean() > 0.01
def run(strategy):
    st = [dict(m=MG(), fc=0, face_run=0, alert=None, read=None) for _ in range(NS)]; proc = psutil.Process(); c0 = proc.cpu_times(); w0 = time.perf_counter(); t0 = w0; costs = []; k = 6
    for r in range(NR):
        if time.perf_counter() < t0 + r / FPS: time.sleep(t0 + r / FPS - time.perf_counter())      # real-time pacing: round r arrives at r/15 s
        for i in range(NS):
            tgt = t0 + r / FPS; f = STREAMS[i][0][r]; do = False
            if strategy == "A": do = (r % k == i % k)
            elif strategy == "B":
                a = st[i]["m"].active(f); st[i]["fc"] = st[i]["fc"] + 1 if a else 0; do = a and st[i]["fc"] % 3 == 1
            else: do = (r % 6 == i % 6)
            if not do: continue
            ts = time.perf_counter(); face, txt = (light_chain if strategy == "C" else heavy_orig)(f); costs.append(ms(ts)); now = time.perf_counter()
            S = st[i]; S["face_run"] = S["face_run"] + 1 if face else 0
            if S["face_run"] >= 2 and S["alert"] is None: S["alert"] = (r, now - (t0 + (STREAMS[i][1]["event"] or r) / FPS), STREAMS[i][1]["event"])
            if txt and S["read"] is None and STREAMS[i][1]["plate"] and txt == STREAMS[i][1]["plate"]: S["read"] = (r, now - (t0 + 30 / FPS))
        if time.perf_counter() < t0 + (r + 1) / FPS and False: pass
    wall = time.perf_counter() - w0; c1 = proc.cpu_times(); cpu = ((c1.user - c0.user) + (c1.system - c0.system)) / wall * 100; lag = wall - NR / FPS
    pers = [(i, st[i]["alert"]) for i in range(NS) if "person" in STREAMS[i][1]["kind"]]; veh = [(i, st[i]["read"]) for i in range(NS) if STREAMS[i][1]["kind"] == "IR vehicle"]
    falses = [i for i in range(NS) if STREAMS[i][1]["event"] is None and st[i]["alert"] is not None] + [i for i in range(NS) if "person" in STREAMS[i][1]["kind"] and st[i]["alert"] is not None and st[i]["alert"][0] < 30]
    return dict(wall_s=wall, stream_frames=len(costs), ms_per_proc=float(np.mean(costs)), cpu_pct=cpu, lag_s=lag, persons_total=len(pers), persons_alerted=sum(a is not None for i, a in pers),
                persons_within_3s=sum(a is not None and a[1] <= 3.0 for i, a in pers), alert_delays=[round(a[1], 2) if a else None for i, a in pers], plates_total=len(veh), plates_ok=sum(a is not None for i, a in veh),
                plate_delays=[round(a[1], 2) if a else None for i, a in veh], false_alerts=len(falses), processed_pct=len(costs) / (NR * NS) * 100)
if __name__ == "__main__":
    out = {}
    # (1) validity of the original modules on night / IR / glare frames (frames where the event is visible)
    val = {}
    for kind in KINDS[:3]:
        i = KINDS.index(kind); fr, meta = STREAMS[i]; idx = list(range(40, 85, 5)); nf = nh = npl = nloc = 0
        for t in idx:
            f = fr[t]; roi = f[100:]; nf += len(V8.SYS.detect(roi)) > 0; g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)); nh += len(HOG.detectMultiScale(g[100:], 0.5, (8, 8), (0, 0), 1.15, 2)[0]) > 0
            cn = P6.localise(f); nloc += cn is not None
        val[kind] = dict(scrfd_rate=nf / len(idx) * 100, hog_rate=nh / len(idx) * 100, plate_localised=nloc / len(idx) * 100)
    out["validity"] = val
    # heavy costs per stream-frame
    f0 = STREAMS[0][0][60]; out["cost_ms"] = dict(original=timeit(lambda: heavy_orig(f0), 5)[0], light=timeit(lambda: light_chain(f0), 5)[0], motion_gate=timeit(lambda: MG().active(f0), 10)[0])
    tot = NS * FPS; out["budget"] = dict(total_fps=tot, per_frame_ms_1core=1000 / tot, per_frame_ms_4core=4000 / tot, original_required_cores=tot * out["cost_ms"]["original"] / 1000, light_required_cores=tot * out["cost_ms"]["light"] / 1000)
    for s in ("A", "B", "C"): out[s] = run(s); print(s, {k: v for k, v in out[s].items()}, flush=True)
    # (2) consent withdrawal: delete templates of subject 3 and show that the identity is no longer matched
    e = V8.emb(cv2.cvtColor(D.orl(3, 8), cv2.COLOR_GRAY2BGR), (.12 * 92, .08 * 112, .88 * 92, .92 * 112), 1.0); before = V8.identify(e)
    keep = V8.LG != 3; V8.EG, V8.LG = V8.EG[keep], V8.LG[keep]; after = V8.identify(e); out["consent"] = dict(before=before[0], after=after[0], templates_left=int(len(V8.LG)))
    save_json("t10.json", out); print(out["validity"], out["cost_ms"], out["budget"], out["consent"])
    tiles = [cv2.resize(STREAMS[i][0][60], (320, 180)) for i in range(4)]; [cv2.putText(t, KINDS[i], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2) for i, t in enumerate(tiles)]
    cv2.imwrite(os.path.join(FIGS, "t10_streams.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
