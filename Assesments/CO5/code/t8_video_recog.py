"""TASK 8 - Video face recognition at the gate: A recognise every frame (k=1, k=3) | B detect + KCF track, recognise once + re-verify | C periodic detection + multi-frame voting.
Detector = SCRFD-500M (Task 3 choice, fits the budget); recogniser = ArcFace-MBF with gallery of 30 enrolled people (Task 4)."""
import cv2, numpy as np, os, time, collections
from common import *
import t3_data as D
from t3_detect import Detectors, iou
W, H, FPS = 960, 540, 25
rng = np.random.default_rng(8)
net = cv2.dnn.readNetFromONNX(os.path.join(MODELS, "buffalo_sc", "w600k_mbf.onnx"))
def emb(img_bgr, box, f):
    x0, y0, x1, y1 = box; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * f, (y1 - y0) * f
    a, b, c, d = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)), int(min(cx + w / 2, img_bgr.shape[1])), int(min(cy + h / 2, img_bgr.shape[0]))
    if c - a < 8 or d - b < 8: return None
    net.setInput(cv2.dnn.blobFromImage(cv2.resize(img_bgr[b:d, a:c], (112, 112)), 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)); e = net.forward().ravel(); return e / np.linalg.norm(e)
GAL = list(range(1, 31)); TH = load_json("t4.json")["res"]["C ArcFace-MBF"]["th_far1"]
def build_gallery(f):
    E, L = [], []
    for s in GAL:
        for k in range(1, 6):
            g = D.orl(s, k); h, w = g.shape; e = emb(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (.12 * w, .08 * h, .88 * w, .92 * h), 1.0); E.append(e); L.append(s)
    return np.array(E), np.array(L)
EG, LG = build_gallery(1.0)
def identify(e):
    d = 1 - EG @ e; i = int(d.argmin()); return (int(LG[i]) if d[i] <= TH else 0), float(d[i])        # 0 = UNKNOWN
# ---------------- synthetic gate scenarios with ground truth ----------------
def face_img(s, t): return D.orl(s, 6 + (t // 20) % 5)            # test photos 6-10 (never used for enrolment); changes every 0.8 s
def scene(kind):
    people = {"S1 single entry": [(3, 100, 700, 200, 110, 200)], "S2 crossing": [(8, 80, 800, 190, 130, 150), (33, 800, 80, 200, 150, 130)],
              "S3 occlusion": [(15, 150, 650, 210, 120, 170)], "S4 low light": [(3, 100, 640, 180, 120, 160), (37, 760, 360, 230, 140, 140)]}[kind]
    n = 125 if kind == "S1 single entry" else 150; bg = D.bg_scene("graf1.png" if kind != "S3 occlusion" else "pic1.png", np.random.default_rng(2), W, H); frames, gts = [], []
    for t in range(n):
        f = bg.copy(); g_t = []
        for (sid, x0, x1, y, s0, s1) in people:
            u = t / (n - 1); x = int(x0 + (x1 - x0) * u); hh = int(s0 + (s1 - s0) * u); ww = int(hh * 92 / 112); yy = int(y + 12 * np.sin(t / 9)); x = int(np.clip(x, 0, W - ww)); 
            D.paste(f, face_img(sid, t), x, yy, ww, hh); g_t.append((sid if sid <= 30 else 0, D.box_of(x, yy, ww, hh)))
        if kind == "S3 occlusion" and 45 <= t <= 95:
            px = int(g_t[0][1][0] + 0.45 * (g_t[0][1][2] - g_t[0][1][0]) - 35); cv2.rectangle(f, (px, 0), (px + 80, H), (70, 70, 80), -1)
        if kind == "S4 low light": f = np.clip((f / 255.0) ** 2.2 * 255 * .5 + rng.normal(0, 6, f.shape), 0, 255).astype(np.uint8)
        frames.append(f); gts.append(g_t)
    return frames, gts
# ---------------- strategies ----------------
class Sys:
    def __init__(s): s.det = Detectors.__new__(Detectors); import onnxruntime as ort; s.det.sc = ort.InferenceSession(os.path.join(MODELS, "buffalo_sc", "det_500m.onnx"), providers=["CPUExecutionProvider"])
    def detect(s, f): return [(b[0], b[1], b[2], b[3]) for b in s.det.run_scrfd(f, 0.5)]
SYS = Sys(); EXP = 1.3
def outputs_A(frames, k):
    outs, ms_f, last = [], [], []
    for t, f in enumerate(frames):
        t0 = time.perf_counter()
        if t % k == 0:
            last = []
            for b in SYS.detect(f):
                e = emb(f, b, EXP); last.append((b, identify(e)[0] if e is not None else 0, None))
        outs.append(list(last)); ms_f.append(ms(t0))
    return outs, ms_f
def outputs_B(frames, det_every=10, reverify=25, min_size=70):
    outs, ms_f, tracks, nid = [], [], [], 0
    for t, f in enumerate(frames):
        t0 = time.perf_counter()
        for tr in tracks:                                                   # tracker update every frame
            ok, bb = tr["trk"].update(f)
            if ok: tr["box"] = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]); tr["lost"] = 0
            else: tr["lost"] += 1
        tracks = [tr for tr in tracks if tr["lost"] < 3]
        if t % det_every == 0:
            for b in SYS.detect(f):
                if not any(iou(b, tr["box"]) > 0.3 for tr in tracks):      # unmatched detection -> new track
                    trk = cv2.TrackerKCF_create(); trk.init(f, (int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]))); tracks.append(dict(trk=trk, box=b, label=None, last_rec=-999, lost=0, id=nid)); nid += 1
        for tr in tracks:                                                   # recognise once at first good frame, then re-verify
            w = tr["box"][2] - tr["box"][0]
            if w >= min_size and (tr["label"] is None or t - tr["last_rec"] >= reverify):
                e = emb(f, tr["box"], EXP)
                if e is not None: tr["label"] = identify(e)[0]; tr["last_rec"] = t
        outs.append([(tr["box"], tr["label"], tr["id"]) for tr in tracks if tr["label"] is not None]); ms_f.append(ms(t0))
    return outs, ms_f
def outputs_C(frames, k=5, N=5):
    outs, ms_f, tracks, nid = [], [], [], 0
    for t, f in enumerate(frames):
        t0 = time.perf_counter()
        if t % k == 0:
            new = []
            for b in SYS.detect(f):
                best = max(tracks, key=lambda tr: iou(b, tr["box"]), default=None)
                if best is not None and iou(b, best["box"]) > 0.3 and best not in [n_ for n_ in new]: tr = best
                else: tr = dict(hist=collections.deque(maxlen=N), id=nid); nid += 1
                tr["box"] = b; e = emb(f, b, EXP)
                if e is not None: tr["hist"].append(identify(e))
                new.append(tr)
            tracks = new
        res = []
        for tr in tracks:
            if tr["hist"]:
                votes = collections.Counter(l for l, d in tr["hist"]); lab, c = votes.most_common(1)[0]
                if len(tr["hist"]) >= 3 and c >= 2: res.append((tr["box"], lab, tr["id"]))
                elif len(tr["hist"]) < 3: res.append((tr["box"], None, tr["id"]))
        outs.append([r for r in res if r[1] is not None]); ms_f.append(ms(t0))
    return outs, ms_f
# ---------------- evaluation ----------------
def evaluate(outs, ms_f, gts):
    n_gt = n_ok = 0; switches = 0; records = []; lat = []; per_person = {}
    for t, (o, g) in enumerate(zip(outs, gts)):
        for pi, (sid, box) in enumerate(g):
            n_gt += 1; cands = [(iou(box, b), l) for b, l, i in o if iou(box, b) > 0.3]
            lab = max(cands)[1] if cands else None
            if lab == sid: n_ok += 1
            per_person.setdefault(pi, []).append((t, lab, sid))
    for pi, seq in per_person.items():
        labs = [l for t, l, s in seq if l is not None]; switches += sum(1 for a, b in zip(labs, labs[1:]) if a != b)
        sid = seq[0][2]; logged, run, prev = [], 0, None                   # attendance policy: 3 consecutive identical enrolled labels, log when different from last logged
        for t, l, s in seq:
            run = run + 1 if (l is not None and l == prev) else 1; prev = l
            if l not in (None, 0) and run >= 3 and (not logged or logged[-1] != l): logged.append(l)
        wrong = [x for x in logged if x != sid]; correct = [x for x in logged if x == sid]
        records.append(dict(gt=sid, logged=logged, wrong=len(wrong), dup=max(0, len(correct) - 1), missed=int(sid != 0 and not correct), false_admit=int(sid == 0 and len(logged) > 0)))
        # decision latency: wall-clock to the first frame at which the label is correct (3 consecutive)
        ok_t = None; r = 0
        for t, l, s in seq:
            r = r + 1 if l == s and (s != 0 or True) else 0
            if r >= 3: ok_t = t - 2; break
        lat.append((sum(ms_f[:ok_t + 3]) / 1000.0) if ok_t is not None else None)
    return dict(frame_acc=n_ok / n_gt * 100, id_switches=switches, wrong_records=sum(r["wrong"] for r in records), dup_records=sum(r["dup"] for r in records),
                missed=sum(r["missed"] for r in records), false_admit=sum(r["false_admit"] for r in records), latency=[x for x in lat if x is not None], never_decided=sum(x is None for x in lat), n_people=len(per_person))
if __name__ == "__main__":
    # tune the crop expansion on enrolment images only (no test leakage): choose f with smallest mean genuine distance
    cal = {}
    for f_ in (1.0, 1.15, 1.3, 1.5):
        ds = []
        for s in range(1, 11):
            sc = D.bg_scene("graf1.png", np.random.default_rng(s), W, H); D.paste(sc, D.orl(s, 1), 300, 200, 150, 182); b = SYS.detect(sc)
            if b: e = emb(sc, b[0], f_); ds.append(float((1 - EG[LG == s] @ e).min()))
        cal[f_] = float(np.mean(ds)) if ds else 9
    EXP = min(cal, key=cal.get); print("crop expansion", cal, "->", EXP, flush=True)
    SCN = ["S1 single entry", "S2 crossing", "S3 occlusion", "S4 low light"]; data = {k: scene(k) for k in SCN}
    strategies = {"A1 every frame": lambda fr: outputs_A(fr, 1), "A3 every 3rd frame": lambda fr: outputs_A(fr, 3), "B detect+track, recognise once": outputs_B, "C periodic detect + voting": outputs_C}
    res = {}; allouts = {}
    for sn, fn in strategies.items():
        res[sn] = {}; allms = []
        for k in SCN:
            fr, gts = data[k]; outs, mf = fn(fr); ev = evaluate(outs, mf, gts); ev["ms_mean"] = float(np.mean(mf)); ev["ms_p95"] = float(np.percentile(mf, 95)); res[sn][k] = ev; allms += mf; allouts[(sn, k)] = outs
        res[sn]["overall"] = dict(ms_mean=float(np.mean(allms)), fps=1000 / float(np.mean(allms)), ms_p95=float(np.percentile(allms, 95)))
        print(sn, {k: (round(v.get("frame_acc", 0), 1), v.get("id_switches")) for k, v in res[sn].items() if k != "overall"}, round(res[sn]["overall"]["ms_mean"], 1), "ms", flush=True)
    # component timing
    f0 = data["S1 single entry"][0][60]; b0 = SYS.detect(f0)
    comp = dict(detect_ms=timeit(lambda: SYS.detect(f0), 8)[0], embed_ms=timeit(lambda: emb(f0, b0[0], EXP), 8)[0] if b0 else None)
    trk = cv2.TrackerKCF_create(); bb = b0[0]; trk.init(f0, (int(bb[0]), int(bb[1]), int(bb[2] - bb[0]), int(bb[3] - bb[1]))); comp["kcf_update_ms"] = timeit(lambda: trk.update(f0), 20)[0]
    comp["full_1080p_detect_ms_est"] = comp["detect_ms"]  # SCRFD letterboxes to 640 so cost is independent of 1080p/540p input; resize cost added below
    big = cv2.resize(f0, (1920, 1080)); comp["detect_1080p_input_ms"] = timeit(lambda: SYS.detect(big), 8)[0]
    save_json("t8.json", dict(res=res, comp=comp, th=TH, exp=EXP, cal=cal, frames={k: len(v[0]) for k, v in data.items()}))
    # annotated video (crossing scenario, three strategies) + montage
    names = ["A1 every frame", "B detect+track, recognise once", "C periodic detect + voting"]; vw = cv2.VideoWriter(os.path.join(VID, "t8_gate_crossing.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (480 * 3, 270)); fr, gts = data["S2 crossing"]; keys = {}
    for t in range(len(fr)):
        row = []
        for n in names:
            a = fr[t].copy()
            for b, l, i in allouts[(n, "S2 crossing")][t]:
                cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0) if l else (0, 0, 255), 3); cv2.putText(a, "UNKNOWN" if l == 0 else f"ID {l}", (int(b[0]), int(b[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(a, n[:30], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3); row.append(cv2.resize(a, (480, 270)))
        img = np.hstack(row); vw.write(img)
        if t in (20, 70, 120): keys[t] = img
    vw.release(); cv2.imwrite(os.path.join(FIGS, "t8_frames.png"), np.vstack(list(keys.values()))); print("done")
