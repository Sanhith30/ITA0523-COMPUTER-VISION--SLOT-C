"""TASK 4 - Face recognition: Eigenfaces vs LBPH vs deep embedding (ArcFace MobileFaceNet, w600k_mbf)."""
import cv2, numpy as np, os, time
from common import *
import t3_data as D
rng = np.random.default_rng(5)
GAL = list(range(1, 31)); UNK = list(range(31, 41))        # 30 enrolled identities, 10 never-enrolled "unknown" people
def load(s, ks): return [D.orl(s, k) for k in ks]
enrol = [(s, g) for s in GAL for g in load(s, range(1, 6))]       # 5 enrolment photos each (as in the brief)
probe = [(s, g) for s in GAL for g in load(s, range(6, 11))]      # genuine probes (other 5 photos)
unk = [(s, g) for s in UNK for g in load(s, range(1, 11))]        # impostor probes
# ---- gate-style conditions applied to probes ----
def cond(g, kind):
    h, w = g.shape; f = g.astype(np.float32)
    if kind == "enrol-style": return g
    if kind == "lighting":                                           # strong side light + gamma
        grad = np.linspace(0.35, 1.5, w, dtype=np.float32)[None, :]; return np.clip(255 * (f / 255) ** 1.4 * grad, 0, 255).astype(np.uint8)
    if kind == "mask": o = g.copy(); cv2.fillPoly(o, [np.array([[int(.1 * w), int(.58 * h)], [int(.9 * w), int(.58 * h)], [int(.85 * w), h], [int(.15 * w), h]])], 185); return o
    if kind == "glasses":
        o = g.copy()
        for cx in (.32, .68): cv2.ellipse(o, (int(cx * w), int(.40 * h)), (int(.16 * w), int(.10 * h)), 0, 0, 360, 10, 4)
        cv2.line(o, (int(.47 * w), int(.40 * h)), (int(.53 * w), int(.40 * h)), 10, 4); return o
    if kind == "low-res+noise":
        s = cv2.resize(g, (w // 3, h // 3), interpolation=cv2.INTER_AREA); s = cv2.resize(s, (w, h)); return np.clip(s + rng.normal(0, 10, s.shape), 0, 255).astype(np.uint8)
CONDS = ["enrol-style", "lighting", "mask", "glasses", "low-res+noise"]
# ---- recognizers ----
def crop(g): h, w = g.shape; return g[int(.08 * h):int(.92 * h), int(.12 * w):int(.88 * w)]
class Eigen:
    name = "A Eigenfaces"
    def fit(self, data): self.m = cv2.face.EigenFaceRecognizer_create(num_components=60); self.m.train([g for _, g in data], np.array([s for s, _ in data]))
    def query(self, g): l, d = self.m.predict(g); return l, float(d)
class LBPH:
    name = "B LBPH"
    def fit(self, data): self.m = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8); self.m.train([g for _, g in data], np.array([s for s, _ in data]))
    def query(self, g): l, d = self.m.predict(g); return l, float(d)
class Deep:
    name = "C ArcFace-MBF"
    def __init__(self): self.net = cv2.dnn.readNetFromONNX(os.path.join(MODELS, "buffalo_sc", "w600k_mbf.onnx"))
    def emb(self, g):
        c = cv2.resize(cv2.cvtColor(crop(g), cv2.COLOR_GRAY2BGR), (112, 112)); self.net.setInput(cv2.dnn.blobFromImage(c, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True))
        e = self.net.forward().ravel(); return e / np.linalg.norm(e)
    def fit(self, data): self.E = np.array([self.emb(g) for _, g in data]); self.L = np.array([s for s, _ in data])
    def query(self, g): d = 1 - self.E @ self.emb(g); i = int(d.argmin()); return int(self.L[i]), float(d[i])
res = {}; roc = {}
for R in (Eigen(), LBPH(), Deep()):
    t0 = time.perf_counter(); R.fit(enrol); fit_s = time.perf_counter() - t0; out = {}
    gen = {c: [(s, R.query(cond(g, c))) for s, g in probe] for c in CONDS}; imp = [R.query(g)[1] for s, g in unk]
    rank1 = {c: float(np.mean([l == s for s, (l, d) in v]) * 100) for c, v in gen.items()}
    base = gen["enrol-style"]; gd = np.array([d if l == s else np.inf for s, (l, d) in base])          # wrong label counts as a rejection
    ths = np.unique(np.concatenate([[d for s, (l, d) in v] for v in gen.values()] + [imp])); ths = np.quantile(ths, np.linspace(0, 1, 400))
    far = np.array([np.mean(np.array(imp) <= t) * 100 for t in ths]); frr = np.array([np.mean(gd > t) * 100 for t in ths]); eer_i = int(np.argmin(abs(far - frr)))
    ok = np.where(far <= 1.0)[0]; t1 = ths[ok[-1]] if len(ok) else ths[0]
    at = {c: float(np.mean([(l != s) or (d > t1) for s, (l, d) in v]) * 100) for c, v in gen.items()}      # FRR per condition at the FAR<=1% threshold
    # timing for one probe vs gallery size (real measurement, replicated templates) and embedding/extract cost
    ptime = {}
    g0 = probe[0][1]
    if isinstance(R, Deep):
        ptime["extract_ms"] = timeit(lambda: R.emb(g0), 15)[0]
        for N in (150, 25000, 100000):
            E = np.random.randn(N, 512).astype(np.float32); E *= (1.0 / np.sqrt(512)); e = R.emb(g0)
            ptime[N] = timeit(lambda: (1 - E @ e).argmin(), 5)[0]
    else:
        for N in (150, 750, 1500):
            data = [(enrol[i % 150][0], np.clip(enrol[i % 150][1].astype(np.int16) + rng.integers(-3, 4, enrol[0][1].shape), 0, 255).astype(np.uint8)) for i in range(N)]
            R2 = type(R)(); R2.fit(data); ptime[N] = timeit(lambda: R2.query(g0), 5)[0]
        ns = np.array([150, 750, 1500]); ts = np.array([ptime[n] for n in ns]); k = np.polyfit(ns, ts, 1); ptime["slope_ms_per_template"] = float(k[0]); ptime["25000_extrap"] = float(np.polyval(k, 25000)); ptime["250000_extrap"] = float(np.polyval(k, 250000))
    f = os.path.join(RES, "model_" + R.name[0] + ".yml")
    size = os.path.getsize(os.path.join(MODELS, "buffalo_sc", "w600k_mbf.onnx")) / 1e6 if isinstance(R, Deep) else (R.m.write(f) or os.path.getsize(f) / 1e6)
    res[R.name] = dict(fit_s=fit_s, rank1=rank1, eer=float((far[eer_i] + frr[eer_i]) / 2), eer_th=float(ths[eer_i]), th_far1=float(t1), frr_at_far1=at, far_at_th1=float(np.mean(np.array(imp) <= t1) * 100),
                       probe_ms=ptime, model_mb=float(size), unknown_rejected_pct=float(100 - np.mean(np.array(imp) <= t1) * 100))
    roc[R.name] = (far.tolist(), frr.tolist()); print(R.name, rank1, "EER", res[R.name]["eer"], flush=True)
save_json("t4.json", dict(res=res, roc=roc, n_gallery=len(enrol), n_genuine=len(probe), n_imp=len(unk)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for n, (far, frr) in roc.items(): ax[0].plot(far, frr, label=n); ax[1].plot(far, 100 - np.array(frr), label=n)
ax[0].set_xlabel("FAR (%)"); ax[0].set_ylabel("FRR (%)"); ax[0].set_title("FAR vs FRR (DET)"); ax[0].set_xlim(0, 100); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xlabel("FAR (%)"); ax[1].set_ylabel("TAR = 100 - FRR (%)"); ax[1].set_title("ROC (enrolment-style probes)"); ax[1].set_xlim(0, 100); ax[1].grid(alpha=.3); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "t4_roc.png"), dpi=130)
