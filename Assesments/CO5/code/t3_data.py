"""Build the labelled face-detection test set: ORL faces (test subjects 21-40) pasted into 1080p scenes with known boxes.
Hard cases are synthesised (see report): small faces, side-pose proxy, mask, glasses, low light, crowd; plus no-face scenes."""
import cv2, numpy as np, os, json, glob
from common import *
BG_TRAIN = ["aloeL.jpg", "building.jpg", "fruits.jpg", "home.jpg", "starry_night.jpg", "board.jpg"]
BG_TEST = ["graf1.png", "pic1.png", "pic3.png", "butterfly.jpg", "HappyFish.jpg"]
NEG_ONLY = ["baboon.jpg"]
def orl(subject, k): return cv2.imread(os.path.join(DATA, "orl", "att", f"s{subject}", f"{k}.pgm"), 0)
def bg_scene(name, rng, W=1920, H=1080):
    im = cv2.imread(os.path.join(DATA, "bg", name)); h, w = im.shape[:2]
    ch = min(h, int(w * 9 / 16)); cw = int(ch * 16 / 9); y0 = rng.integers(0, h - ch + 1); x0 = rng.integers(0, w - cw + 1)
    return cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_CUBIC)
def box_of(x, y, w, h): return [x + 0.12 * w, y + 0.08 * h, x + 0.88 * w, y + 0.92 * h]
def variant(g, kind):
    g = g.copy(); h, w = g.shape
    if kind == "mask": cv2.fillPoly(g, [np.array([[int(.12 * w), int(.60 * h)], [int(.88 * w), int(.60 * h)], [int(.80 * w), int(.97 * h)], [int(.20 * w), int(.97 * h)]])], 190)
    if kind == "glasses":
        for cx in (.32, .68): cv2.ellipse(g, (int(cx * w), int(.40 * h)), (int(.15 * w), int(.09 * h)), 0, 0, 360, 15, 3)
        cv2.line(g, (int(.47 * w), int(.40 * h)), (int(.53 * w), int(.40 * h)), 15, 3)
    if kind == "side":                                   # yaw proxy: squeeze + shear (not a true rotation in 3-D)
        M = np.float32([[0.6, 0.12, 0.2 * w], [0, 1, 0]]); g = cv2.warpAffine(g, M, (w, h), borderValue=int(g[:5].mean()))
    return g
def paste(scene, g, x, y, w, h):
    f = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC); a = np.ones((h, w), np.float32)
    k = max(3, (min(w, h) // 8) | 1); a[:k // 2 + 1] = a[-(k // 2 + 1):] = 0; a[:, :k // 2 + 1] = a[:, -(k // 2 + 1):] = 0
    a = cv2.GaussianBlur(a, (k, k), 0)[..., None]; roi = scene[y:y + h, x:x + w].astype(np.float32)
    scene[y:y + h, x:x + w] = (a * cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) + (1 - a) * roi).astype(np.uint8)
def build():
    rng = np.random.default_rng(11); outdir = os.path.join(DATA, "scenes"); os.makedirs(outdir, exist_ok=True)
    spec = [("frontal", 8, 1, (170, 260)), ("small", 8, 2, (44, 70)), ("side", 6, 1, (150, 230)), ("mask", 6, 1, (150, 230)),
            ("glasses", 6, 1, (150, 230)), ("lowlight", 8, 1, (140, 230)), ("crowd", 4, 3, (90, 150)), ("noface", 8, 0, (0, 0))]
    meta = []; subj = list(range(21, 41)); n = 0
    for cat, count, nf, (s0, s1) in spec:
        for i in range(count):
            bgn = (NEG_ONLY + BG_TEST)[i % 6] if cat == "noface" else BG_TEST[(n + i) % len(BG_TEST)]
            sc = bg_scene(bgn, rng); boxes = []
            for _ in range(nf):
                for _try in range(50):
                    hh = int(rng.integers(s0, s1 + 1)); ww = int(hh * 92 / 112); x = int(rng.integers(20, 1920 - ww - 20)); y = int(rng.integers(20, 1080 - hh - 20))
                    if all(x > b[2] + 40 or x + ww < b[0] - 40 or y > b[3] + 40 or y + hh < b[1] - 40 for b in boxes): break
                g = orl(int(rng.choice(subj)), int(rng.integers(1, 11))); g = variant(g, cat if cat in ("mask", "glasses", "side") else "none")
                paste(sc, g, x, y, ww, hh); boxes.append(box_of(x, y, ww, hh))
            if cat == "lowlight":
                sc = np.clip((sc.astype(np.float32) / 255) ** 2.2 * 255 * 0.5 + rng.normal(0, 4, sc.shape), 0, 255).astype(np.uint8)
            name = f"{cat}_{i:02d}.jpg"; cv2.imwrite(os.path.join(outdir, name), sc, [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta.append(dict(file=name, cat=cat, boxes=[[float(v) for v in b] for b in boxes]))
        n += count
    json.dump(meta, open(os.path.join(outdir, "gt.json"), "w")); print(len(meta), "scenes", sum(len(m["boxes"]) for m in meta), "faces")
if __name__ == "__main__": build()
