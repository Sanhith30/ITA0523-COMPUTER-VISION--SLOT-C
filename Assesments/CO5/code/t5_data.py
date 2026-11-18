"""TASK 5 data: JAFFE (213 posed images, 10 subjects) -> 3 classes (happy / neutral / unhappy); subject-independent split + candid-proxy test set."""
import cv2, numpy as np, os, glob
from common import *
rng = np.random.default_rng(21)
files = sorted(glob.glob(os.path.join(DATA, "jaffe", "*", "*.tiff")))
MAP = {"HA": 0, "NE": 1, "AN": 2, "DI": 2, "FE": 2, "SA": 2}          # SU (surprise) is excluded
X, y, subj, code = [], [], [], []
for f in files:
    b = os.path.basename(f).split("."); s, e = b[0], b[1][:2]
    if e not in MAP: continue
    g = cv2.imread(f, 0); X.append(cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)); y.append(MAP[e]); subj.append(s); code.append(e)
X = np.array(X); y = np.array(y); subj = np.array(subj); code = np.array(code)
TEST_S = ["UY", "YM", "TM"]; te = np.isin(subj, TEST_S); tr = ~te
def candid(g):                                           # proxy for candid kiosk images (rotation, scale, blur, low light, noise, occlusion)
    h, w = g.shape; M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-18, 18), rng.uniform(.85, 1.05)); o = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    o = cv2.GaussianBlur(o, (5, 5), rng.uniform(.5, 2.0)).astype(np.float32); o = 255 * (o / 255) ** rng.uniform(1.0, 2.0) * rng.uniform(.5, 1.0)
    o += rng.normal(0, 8, o.shape); o = np.clip(o, 0, 255).astype(np.uint8)
    if rng.random() < .35: x0 = rng.integers(0, 60); cv2.rectangle(o, (x0, 70), (x0 + 60, 128), int(rng.integers(40, 200)), -1)   # hand / drink in front of the mouth
    return o
Xc = np.array([candid(g) for g in X[te]])
np.savez(os.path.join(DATA, "t5.npz"), X=X, y=y, subj=subj, code=code, tr=tr, te=te, Xc=Xc)
print("images", len(X), "train", tr.sum(), "test", te.sum(), "class counts all", np.bincount(y), "train", np.bincount(y[tr]), "test", np.bincount(y[te]))
