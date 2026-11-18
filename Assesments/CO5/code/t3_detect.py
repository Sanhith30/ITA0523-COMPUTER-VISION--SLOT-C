"""TASK 3 - Face detection: Haar cascade vs HOG+linear SVM (trained here) vs SSD (res10) + a current detector (SCRFD/RetinaFace family)."""
import cv2, numpy as np, os, json, time, sys
import onnxruntime as ort
from sklearn.svm import LinearSVC
from common import *
import t3_data as D
WIN = (64, 80)

def hog_obj(): return cv2.HOGDescriptor(WIN, (16, 16), (8, 8), (8, 8), 9)
def crop_face(g, rng=None, jitter=False):
    h, w = g.shape; x0, y0, x1, y1 = .12 * w, .08 * h, .88 * w, .92 * h
    if jitter:
        s = rng.uniform(.93, 1.07); cx, cy = (x0 + x1) / 2 + rng.uniform(-3, 3), (y0 + y1) / 2 + rng.uniform(-3, 3)
        bw, bh = (x1 - x0) * s, (y1 - y0) * s; x0, x1, y0, y1 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
    M = np.float32([[WIN[0] / (x1 - x0), 0, -x0 * WIN[0] / (x1 - x0)], [0, WIN[1] / (y1 - y0), -y0 * WIN[1] / (y1 - y0)]])
    return cv2.warpAffine(g, M, WIN, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
def train_hog(seed=3):
    rng = np.random.default_rng(seed); hog = hog_obj(); pos = []
    for s in range(1, 21):
        for k in range(1, 11):
            g = D.orl(s, k)
            for j in range(4):
                c = crop_face(g, rng, jitter=j > 0); c = np.clip(c.astype(np.float32) * rng.uniform(.7, 1.3) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
                pos.append(hog.compute(c).ravel()); pos.append(hog.compute(cv2.flip(c, 1)).ravel())
    bgs = [D.bg_scene(n, rng, 960, 540) for n in D.BG_TRAIN for _ in range(2)]
    neg = []
    for im in bgs:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for _ in range(350):
            sz = int(rng.integers(50, 300)); x = int(rng.integers(0, 960 - sz)); y = int(rng.integers(0, 540 - sz))
            neg.append(hog.compute(cv2.resize(g[y:y + sz, x:x + sz], WIN, interpolation=cv2.INTER_AREA)).ravel())
    X = np.array(pos + neg, np.float32); y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LinearSVC(C=0.01, max_iter=5000).fit(X, y)
    info = dict(n_pos=len(pos), n_neg=len(neg))
    for rnd in range(2):                                           # hard-negative mining
        hog.setSVMDetector(np.append(clf.coef_.ravel(), clf.intercept_).astype(np.float32)); hard = []
        for im in bgs:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); rects, _ = hog.detectMultiScale(g, -0.5, (8, 8), (0, 0), 1.15, 0)
            for (x, y0, w, h) in rects[:300]:
                if w > 20 and h > 20: hard.append(hog.compute(cv2.resize(g[max(y0, 0):y0 + h, max(x, 0):x + w], WIN, interpolation=cv2.INTER_AREA)).ravel())
        info[f"hard_round{rnd + 1}"] = len(hard)
        if not hard: break
        X = np.vstack([X, np.array(hard, np.float32)]); y = np.concatenate([y, np.zeros(len(hard), int)]); clf = LinearSVC(C=0.01, max_iter=5000).fit(X, y)
    hog.setSVMDetector(np.append(clf.coef_.ravel(), clf.intercept_).astype(np.float32)); info["train_acc"] = float(clf.score(X, y))
    return hog, info

class Detectors:
    def __init__(self):
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.ssd = cv2.dnn.readNetFromCaffe(os.path.join(MODELS, "deploy.prototxt"), os.path.join(MODELS, "res10.caffemodel"))
        self.sc = ort.InferenceSession(os.path.join(MODELS, "buffalo_sc", "det_500m.onnx"), providers=["CPUExecutionProvider"])
        self.hog, self.hog_info = train_hog(); self.hog_th = 0.5
    def run_haar(self, img, scale=1.0):
        im = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)); r = self.haar.detectMultiScale(g, 1.1, 5, minSize=(30, 30) if scale == 1 else (24, 24))
        return [(x / scale, y / scale, (x + w) / scale, (y + h) / scale, 1.0) for x, y, w, h in r]
    def run_hog(self, img, scale=1.0):
        im = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); r, w = self.hog.detectMultiScale(g, self.hog_th, (8, 8), (0, 0), 1.15, 2)
        return [(x / scale, y / scale, (x + ww) / scale, (y + hh) / scale, float(s)) for (x, y, ww, hh), s in zip(r, np.ravel(w))] if len(r) else []
    def run_ssd(self, img, size=(300, 300), thr=0.5):
        H, W = img.shape[:2]; self.ssd.setInput(cv2.dnn.blobFromImage(img, 1.0, size, (104, 177, 123))); d = self.ssd.forward()[0, 0]
        return [(d_[3] * W, d_[4] * H, d_[5] * W, d_[6] * H, float(d_[2])) for d_ in d if d_[2] > thr]
    def run_scrfd(self, img, thr=0.5, size=640):
        h, w = img.shape[:2]; r = min(size / h, size / w); nh, nw = int(h * r), int(w * r)
        canvas = np.zeros((size, size, 3), np.uint8); canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (size, size), (127.5,) * 3, swapRB=True)
        outs = [o.reshape(-1, o.shape[-1]) for o in self.sc.run(None, {self.sc.get_inputs()[0].name: blob})]; B, S = [], []
        for stride in (8, 16, 32):
            fm = size // stride; n = fm * fm * 2
            sc = [o for o in outs if o.shape == (n, 1)][0][:, 0]; bb = [o for o in outs if o.shape == (n, 4)][0] * stride
            ys, xs = np.mgrid[:fm, :fm]; c = np.repeat(np.stack([xs, ys], -1).reshape(-1, 2) * stride, 2, axis=0); k = sc > thr
            B.append(np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1], c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], 1)[k] / r); S.append(sc[k])
        B, S = np.concatenate(B), np.concatenate(S)
        if not len(S): return []
        idx = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in B.tolist()], S.tolist(), thr, 0.4)
        return [(*B[i], float(S[i])) for i in np.array(idx).ravel()]

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
def match(dets, gts, thr=0.3):
    used, tp, fp = set(), 0, 0
    for d in sorted(dets, key=lambda d: -d[4]):
        best, bj = 0, -1
        for j, g in enumerate(gts):
            if j in used: continue
            v = iou(d, g)
            if v > best: best, bj = v, j
        if best >= thr: used.add(bj); tp += 1
        else: fp += 1
    return tp, fp, used

if __name__ == "__main__":
    t0 = time.time(); det = Detectors(); print("trained HOG", det.hog_info, round(time.time() - t0), "s", flush=True)
    meta = json.load(open(os.path.join(DATA, "scenes", "gt.json")))
    # --- validation of the HOG hit threshold on scenes built from TRAIN backgrounds / TRAIN subjects (no test leakage) ---
    rng = np.random.default_rng(99); val = []
    for i in range(8):
        sc = D.bg_scene(D.BG_TRAIN[i % 6], rng); hh = int(rng.integers(150, 250)); ww = int(hh * 92 / 112); x = int(rng.integers(20, 1700)); y = int(rng.integers(20, 800))
        D.paste(sc, D.orl(int(rng.integers(1, 21)), int(rng.integers(1, 11))), x, y, ww, hh); val.append((sc, [D.box_of(x, y, ww, hh)]))
    best = None; vth = {}
    for th in (0.0, 0.5, 1.0, 1.5):
        det.hog_th = th; tp = fp = 0
        for sc, g in val:
            a, b, _ = match(det.run_hog(sc, 0.5), g); tp += a; fp += b
        f1 = 2 * tp / (2 * tp + fp + (len(val) - tp)); vth[th] = dict(tp=tp, fp=fp, f1=f1)
        if best is None or f1 > best[1]: best = (th, f1)
    det.hog_th = best[0]; print("HOG validation", vth, "chosen", best, flush=True)
    cfgs = {"Haar @1080p": lambda im: det.run_haar(im, 1.0), "Haar @540p": lambda im: det.run_haar(im, 0.5),
            "HOG+SVM @1080p": lambda im: det.run_hog(im, 1.0), "HOG+SVM @540p": lambda im: det.run_hog(im, 0.5),
            "SSD 960x540 input": lambda im: det.run_ssd(im, (960, 540)), "SSD 300x300 input": lambda im: det.run_ssd(im, (300, 300)),
            "SCRFD 640 (current)": lambda im: det.run_scrfd(im)}
    cats = sorted({m["cat"] for m in meta}); res = {k: dict(tp=0, fp=0, gt=0, ms=[], cat={c: [0, 0] for c in cats}) for k in cfgs}; vis = {}
    for m in meta:
        im = cv2.imread(os.path.join(DATA, "scenes", m["file"]))
        for k, fn in cfgs.items():
            t = time.perf_counter(); dets = fn(im); dt = ms(t); tp, fp, used = match(dets, m["boxes"])
            r = res[k]; r["tp"] += tp; r["fp"] += fp; r["gt"] += len(m["boxes"]); r["ms"].append(dt); r["cat"][m["cat"]][0] += tp; r["cat"][m["cat"]][1] += len(m["boxes"])
            if m["file"] in ("mask_01.jpg", "side_01.jpg", "lowlight_02.jpg"): vis.setdefault(m["file"], {})[k] = dets
        print(m["file"], flush=True)
    out = {}
    for k, r in res.items():
        P = r["tp"] / max(r["tp"] + r["fp"], 1); R = r["tp"] / r["gt"]
        out[k] = dict(precision=P * 100, recall=R * 100, fp_per_img=r["fp"] / len(meta), ms_mean=float(np.mean(r["ms"])), ms_p95=float(np.percentile(r["ms"], 95)),
                      cat_recall={c: (v[0] / v[1] * 100 if v[1] else None) for c, v in r["cat"].items()}, tp=r["tp"], fp=r["fp"], gt=r["gt"])
    # ROI timing (gate-line ROI 640x360 crop and 960x540 whole) on 8 scenes
    roi_t = {}
    for k, fn in cfgs.items():
        ts = []
        for m in meta[:8]:
            im = cv2.imread(os.path.join(DATA, "scenes", m["file"]))[360:720, 640:1280].copy(); t = time.perf_counter(); fn(im); ts.append(ms(t))
        roi_t[k] = float(np.mean(ts))
    save_json("t3.json", dict(results=out, roi_ms=roi_t, hog=det.hog_info, hog_val=vth, hog_th=det.hog_th, n_scenes=len(meta), n_faces=sum(len(m["boxes"]) for m in meta)))
    # annotated example figure
    rows = []
    for f, d in vis.items():
        im = cv2.imread(os.path.join(DATA, "scenes", f)); tiles = []
        for k in ("Haar @540p", "HOG+SVM @540p", "SSD 960x540 input", "SCRFD 640 (current)"):
            a = im.copy()
            for (x0, y0, x1, y1, s) in d[k]: cv2.rectangle(a, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 6)
            cv2.putText(a, k, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6); tiles.append(cv2.resize(a, (400, 225)))
        rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(FIGS, "t3_detections.png"), np.vstack(rows)); print("done")
