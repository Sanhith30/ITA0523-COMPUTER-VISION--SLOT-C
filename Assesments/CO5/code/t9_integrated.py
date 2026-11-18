"""TASK 9 - Integrated campus application (gate face recognition + plate OCR + corridor intrusion) in three architectures, all fed with the same 25 fps streams in real time."""
import cv2, numpy as np, os, time, threading, queue, collections, inspect, psutil
from common import *
import t8_video_recog as V8
import t6_ocr as P6
import t7_motion as T7
V8.EXP = 1.0
FPS, NR = 25, 200
# ---------------- identical input streams ----------------
g1, gt1 = V8.scene("S2 crossing"); g2, gt2 = V8.scene("S4 low light"); GATE = g1 + g2[:50]; GTS = gt1 + gt2[:50]
rng = np.random.default_rng(90)
bg = np.clip(np.full((450, 800, 3), 85, np.float32) + rng.normal(0, 8, (450, 800, 3)), 0, 255).astype(np.uint8); cv2.rectangle(bg, (120, 120), (680, 330), (45, 45, 55), -1)
VEH = [(20, 50, "clean"), (80, 110, "skew"), (140, 170, "blur")]; plates = [P6.rand_plate() for _ in VEH]; vscene = [P6.make_scene(p, c) for p, (a, b, c) in zip(plates, VEH)]
PLATE = []
for t in range(NR):
    f = bg.copy()
    for (a, b, c), sc in zip(VEH, vscene):
        if a <= t <= b: f = np.roll(sc, (int(rng.integers(-2, 3)), int(rng.integers(-3, 4))), (0, 1))
    PLATE.append(np.clip(f.astype(np.float32) + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8))
cfr, cen, cxs = T7.render(T7.SCN["S4 fast intruder + tree"], seed=3); CORR = cfr[200:400]; ENTER = cen - 200
# ---------------- module wrappers ----------------
def gate_proc(f):
    out = []
    for b in V8.SYS.detect(f):
        e = V8.emb(f, b, 1.0); out.append((b, V8.identify(e)[0] if e is not None else 0, None))
    return out
def plate_proc(f):
    cn = P6.localise(f)
    if cn is None: return None
    return P6.ocr_C(P6.rectify(f, cn))[0]
class Corr:
    def __init__(s): s.m = T7.B(-1); s.cnt = 0; s.last = -999; s.t = 0
    def step(s, g):
        m = s.m.step(g); bl = T7.blobs(m, 60); fg = (m > 0).mean(); inr = [b for b in bl if cv2.pointPolygonTest(T7.ROI, (b[0], b[1]), False) >= 0]
        s.cnt = s.cnt + 1 if (inr and fg < 0.4) else 0; s.t += 1; al = s.cnt >= 5 and s.t - s.last > 75
        if al: s.last = s.t
        return al
class Gate:                                         # cheap motion gate (MOG2 on a 1/4-size frame)
    def __init__(s): s.m = cv2.createBackgroundSubtractorMOG2(200, 25, False)
    def active(s, f): m = s.m.apply(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4), interpolation=cv2.INTER_AREA)); return (m > 200).mean() > 0.004
# ---------------- architectures ----------------
def arch_A(rec):
    cs = Corr(); t0 = time.perf_counter()
    for r in range(NR):
        tgt = t0 + r / FPS
        if time.perf_counter() < tgt: time.sleep(tgt - time.perf_counter())
        rec["gate"][r] = gate_proc(GATE[r]); p = plate_proc(PLATE[r]); rec["plate"][r] = p
        if cs.step(CORR[r]): rec["alerts"].append((r, time.perf_counter()))
        rec["fin"][r] = time.perf_counter(); rec["tgt"][r] = tgt
    return t0
def arch_B(rec):
    cs = Corr(); gg, gp = Gate(), Gate(); t0 = time.perf_counter(); k = 0
    for r in range(NR):
        tgt = t0 + r / FPS
        if time.perf_counter() < tgt: time.sleep(tgt - time.perf_counter())
        a = gg.active(GATE[r]); rec["gate"][r] = gate_proc(GATE[r]) if a else []
        ap = gp.active(PLATE[r]); k = k + 1 if ap else 0; rec["plate"][r] = plate_proc(PLATE[r]) if (ap and k % 4 == 1) else None
        if cs.step(CORR[r]): rec["alerts"].append((r, time.perf_counter()))
        rec["fin"][r] = time.perf_counter(); rec["tgt"][r] = tgt
    return t0
def arch_C(rec):
    qs = {k: queue.Queue(maxsize=2) for k in ("gate", "plate", "corr")}; done = threading.Event(); drops = collections.Counter()
    def worker(name):
        cs = Corr() if name == "corr" else None
        while True:
            try: r, f = qs[name].get(timeout=0.3)
            except queue.Empty:
                if done.is_set(): return
                continue
            if name == "gate": rec["gate"][r] = gate_proc(f)
            elif name == "plate": rec["plate"][r] = plate_proc(f)
            else:
                if cs.step(f): rec["alerts"].append((r, time.perf_counter()))
            rec["fin_c"][name][r] = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(n,), daemon=True) for n in qs]; [t.start() for t in ths]; t0 = time.perf_counter()
    for r in range(NR):
        tgt = t0 + r / FPS
        if time.perf_counter() < tgt: time.sleep(tgt - time.perf_counter())
        rec["tgt"][r] = tgt
        for n, src in (("gate", GATE), ("plate", PLATE), ("corr", CORR)):
            try: qs[n].put_nowait((r, src[r]))
            except queue.Full:
                try: qs[n].get_nowait(); drops[n] += 1
                except queue.Empty: pass
                qs[n].put_nowait((r, src[r]))
    time.sleep(1.5); done.set(); [t.join() for t in ths]; rec["drops"] = dict(drops); return t0
def evaluate_run(rec, t0, wall, cpu):
    gate = [None] * NR; last = []
    for r in range(NR):
        if rec["gate"][r] is not None: last = rec["gate"][r]
        gate[r] = list(last)
    e1 = V8.evaluate(gate[:150], [0] * 150, GTS[:150]); e2 = V8.evaluate(gate[150:], [0] * 50, GTS[150:])
    reads = []
    for (a, b, c), gt in zip(VEH, plates):
        rr = [rec["plate"][r] for r in range(a, b + 1) if rec["plate"][r]]; pred = collections.Counter(rr).most_common(1)[0][0] if rr else ""
        reads.append(dict(cond=c, gt=gt, pred=pred, ok=pred == gt, char_acc=max(0, 1 - P6.lev_ops(pred, gt)[0] / len(gt))))
    al = [(r, t) for r, t in rec["alerts"] if r >= ENTER - 5]; fa = len([1 for r, t in rec["alerts"] if r < ENTER - 5])
    lat = (al[0][1] - (t0 + ENTER / FPS)) if al else None
    fin = np.array([rec["fin"][r] - rec["tgt"][r] for r in range(NR) if rec["fin"][r]]) * 1000 if any(rec["fin"]) else None
    return dict(gate_crossing_acc=e1["frame_acc"], gate_lowlight_acc=e2["frame_acc"], id_switches=e1["id_switches"] + e2["id_switches"], plate_reads=reads, plate_ok=sum(x["ok"] for x in reads),
                plate_char_acc=float(np.mean([x["char_acc"] for x in reads]) * 100), alert_latency_s=lat, corridor_false_alarms=fa, wall_s=wall, rounds_per_s=NR / wall, cpu_pct=cpu,
                frame_age_ms_mean=float(fin.mean()) if fin is not None else None, frame_age_ms_p95=float(np.percentile(fin, 95)) if fin is not None else None, frame_age_ms_last=float(fin[-1]) if fin is not None else None, drops=rec.get("drops", {}))
if __name__ == "__main__":
    out = {}
    for name, fn in (("A sequential", arch_A), ("B event-triggered", arch_B), ("C multithreaded", arch_C)):
        rec = dict(gate=[None] * NR, plate=[None] * NR, alerts=[], fin=[0] * NR, tgt=[0] * NR, fin_c={"gate": {}, "plate": {}, "corr": {}}); p = psutil.Process(); c0 = p.cpu_times(); w0 = time.perf_counter()
        t0 = fn(rec); wall = time.perf_counter() - w0; c1 = p.cpu_times(); cpu = ((c1.user - c0.user) + (c1.system - c0.system)) / wall * 100
        if name.startswith("C"):
            ages = [(rec["fin_c"][n][r] - rec["tgt"][r]) * 1000 for n in rec["fin_c"] for r in rec["fin_c"][n]]; ev = evaluate_run(rec, t0, wall, cpu)
            ev["frame_age_ms_mean"] = float(np.mean(ages)); ev["frame_age_ms_p95"] = float(np.percentile(ages, 95)); ev["frame_age_ms_last"] = None
        else: ev = evaluate_run(rec, t0, wall, cpu)
        ev["loc"] = len(inspect.getsource(fn).splitlines()) + (len(inspect.getsource(Gate).splitlines()) if "B" in name[:1] else 0); out[name] = ev
        print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ev.items() if k != "plate_reads"}, flush=True)
    # per-module timings for the 40 ms budget allocation
    fg = GATE[60]; fp = PLATE[30]; fc = CORR[100]; cs = Corr(); gg = Gate()
    bud = dict(capture_decode=6.4, motion_gate=timeit(lambda: gg.active(fg), 10)[0], face_detect=timeit(lambda: V8.SYS.detect(fg), 8)[0], face_embed_match=timeit(lambda: V8.emb(fg, V8.SYS.detect(fg)[0], 1.0), 8)[0],
               plate_localise=timeit(lambda: P6.localise(fp), 10)[0], plate_ocr=timeit(lambda: plate_proc(fp), 8)[0], corridor_mog2=timeit(lambda: cs.step(fc), 20)[0])
    save_json("t9.json", dict(arch=out, budget=bud, enter=ENTER, plates=plates)); print(bud)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=170); ax.set_xlim(0, 110); ax.set_ylim(0, 54); ax.axis("off")
    def box(x, y, w, h, t, b, fc): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#1F3864", lw=1.2)); ax.text(x + w / 2, y + h - 1.6, t, ha="center", va="top", fontsize=7.6, fontweight="bold", color="#1F3864"); ax.text(x + w / 2, y + h - 5, b, ha="center", va="top", fontsize=6.4, linespacing=1.3)
    def ar(a, b, c, d): ax.add_patch(FancyArrowPatch((a, b), (c, d), arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#1F3864"))
    ax.text(55, 52.3, "Integrated campus gate application - block diagram (40 ms frame budget at 25 fps)", ha="center", fontsize=9.5, fontweight="bold")
    box(1, 38, 16, 11, "INPUT", "Gate cam / barrier cam /\ncorridor cam (RTSP)\nstill images\n~6 ms decode", "#F2F2F2")
    box(21, 38, 18, 11, "PRE-PROCESS", "latest-frame reader\nROI crop, resize,\nMOG2 motion gate\n~1 ms", "#DDEBF7")
    box(43, 44, 21, 7, "DETECT: faces", "SCRFD-500M, 640 letterbox  ~25 ms", "#E2EFDA"); box(43, 35, 21, 7, "DETECT: plate", "contour + perspective  ~3 ms", "#E2EFDA"); box(43, 26, 21, 7, "DETECT: motion", "MOG2 + ROI rule  ~1 ms", "#E2EFDA")
    box(68, 44, 21, 7, "RECOGNISE: face", "ArcFace-MBF + gallery  ~20 ms", "#FFF2CC"); box(68, 35, 21, 7, "RECOGNISE: plate", "deep OCR (CRNN)  ~25 ms", "#FFF2CC"); box(68, 26, 21, 7, "EXPRESSION (optional)", "pre-trained ONNX  ~16 ms", "#FFF2CC")
    box(21, 12, 22, 10, "DECISION", "threshold + 3-frame vote,\nregex check on plates,\nalert rules (5 frames)", "#FCE4D6"); box(47, 12, 20, 10, "ANNOTATE", "draw on a COPY of frame\nboxes, IDs, tripwire", "#FCE4D6")
    box(71, 12, 18, 10, "ALERTS + LOGGING", "event log, templates only;\nretention 30 days", "#FCE4D6"); box(93, 12, 15, 10, "SECURITY DESK", "alert in < 2 s\nhuman review", "#F2F2F2")
    for a in ((17.5, 43.5, 20.5, 43.5), (39.5, 43.5, 42.5, 47.5), (39.5, 43.5, 42.5, 38.5), (39.5, 43.5, 42.5, 29.5), (64.5, 47.5, 67.5, 47.5), (64.5, 38.5, 67.5, 38.5), (64.5, 29.5, 67.5, 29.5), (78, 25.5, 33, 22.6), (54, 25.5, 33, 22.6), (43.5, 17, 46.5, 17), (67.5, 17, 70.5, 17), (89.5, 17, 92.5, 17)): ar(*a)
    ax.text(3, 5, "Budget (measured, 1 core): decode 6.4 + gate 1 + face 25 + embed 20 + plate 28 + corridor 1 = 81 ms if all modules run on every frame -> event gating / threads are required", fontsize=7)
    plt.savefig(os.path.join(FIGS, "t9_block.png"), bbox_inches="tight"); print("done")
