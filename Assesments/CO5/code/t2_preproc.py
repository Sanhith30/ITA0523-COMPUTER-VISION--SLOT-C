"""TASK 2 - Pre-processing for plates and corridor frames: Otsu vs adaptive Gaussian vs CLAHE(+Otsu)."""
import itertools, cv2, numpy as np, pytesseract
from common import *
rng = np.random.default_rng(7)
FONT = cv2.FONT_HERSHEY_DUPLEX
PLATES = ["TN09AB1234","KA05MN4821","MH12DE1433","DL3CAF0001","TN22BX7305","AP39CR2468","KL07BW9150","TS09EK6082","GJ01HH7786","UP32LM5019","WB74AC3347","RJ14CV8652"]

def render_plate(text):
    H, W = 120, 520; img = np.full((H, W, 3), 255, np.uint8); mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(img, (4, 4), (W - 5, H - 5), (0, 0, 0), 3)
    (tw, th), _ = cv2.getTextSize(text, FONT, 2.3, 5); x, y = (W - tw) // 2, (H + th) // 2
    cv2.putText(img, text, (x, y), FONT, 2.3, (0, 0, 0), 5, cv2.LINE_AA); cv2.putText(mask, text, (x, y), FONT, 2.3, 255, 5, cv2.LINE_AA)
    return img, mask

def degrade(img, cond):
    f = img.astype(np.float32) / 255; H, W = f.shape[:2]; yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if cond == "day":                                  # bright, mild shading
        out = f * (0.80 + 0.2 * xx / W)[..., None]; sig = 0.02
    elif cond == "night":                              # dark, uneven street-lamp light, noisy
        out = f * (0.07 + 0.16 * np.exp(-((xx - W * rng.uniform(.2, .6)) ** 2) / (2 * (W * .22) ** 2)))[..., None]; sig = 0.03
    else:                                              # headlight glare: additive veiling light saturating one region
        cx, cy = W * rng.uniform(.3, .7), H * rng.uniform(.3, .7)
        veil = 0.95 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (W * .17) ** 2)))
        out = f * 0.75 + veil[..., None]; sig = 0.02
    out = cv2.GaussianBlur(out, (3, 3), 0.8) + rng.normal(0, sig, out.shape)
    return np.clip(out * 255, 0, 255).astype(np.uint8)

# ---- the three pre-processing techniques (each returns a binary image: black text on white) ----
def prep_A(g):                                         # global Otsu
    b = cv2.GaussianBlur(g, (5, 5), 0); return cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def prep_B(g, block=31, C=10):                         # adaptive Gaussian
    b = cv2.GaussianBlur(g, (5, 5), 0); return cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)
def prep_C(g, clip=2.0, tile=8, blur_first=True):      # CLAHE then Otsu
    if blur_first: g = cv2.GaussianBlur(g, (5, 5), 0)
    e = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)
    if not blur_first: e = cv2.GaussianBlur(e, (5, 5), 0)
    return cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
TECH = {"A Otsu": prep_A, "B Adaptive": prep_B, "C CLAHE+Otsu": prep_C}

def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]
def ocr(bin_img):
    im = cv2.copyMakeBorder(bin_img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    t = pytesseract.image_to_string(im, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(c for c in t if c.isalnum())
def char_acc(pred, gt): return max(0.0, 1 - lev(pred, gt) / len(gt))
def f1_mask(bin_img, gt):
    m = np.s_[14:-14, 14:-14]; p = (bin_img[m] == 0); g = gt[m] > 0
    tp = (p & g).sum(); pr = tp / max(p.sum(), 1); rc = tp / max(g.sum(), 1); return float(2 * pr * rc / max(pr + rc, 1e-9))

if __name__ == "__main__":
    conds = ["day"] * 4 + ["night"] * 4 + ["glare"] * 4
    items = []
    for t, c in zip(PLATES, conds):
        im, m = render_plate(t); items.append((t, c, degrade(im, c), m))
    rows = []
    for t, c, im, m in items:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for n, fn in TECH.items():
            b = fn(g); p = ocr(b); rows.append(dict(plate=t, cond=c, tech=n, ocr=p, char_acc=char_acc(p, t), exact=p == t, f1=f1_mask(b, m)))
    summ = {}
    for n in TECH:
        for c in ["day", "night", "glare", "all"]:
            r = [x for x in rows if x["tech"] == n and (c == "all" or x["cond"] == c)]
            summ[f"{n}|{c}"] = dict(char_acc=float(np.mean([x["char_acc"] for x in r]) * 100), plate_acc=float(np.mean([x["exact"] for x in r]) * 100), f1=float(np.mean([x["f1"] for x in r]) * 100), n=len(r))
    # timing: plate crop (520x120) and full 1080p grey frame
    big = cv2.resize(cv2.cvtColor(items[5][2], cv2.COLOR_BGR2GRAY), (1920, 1080)); small = cv2.cvtColor(items[5][2], cv2.COLOR_BGR2GRAY)
    timing = {n: dict(plate_ms=timeit(lambda: fn(small), 30)[0], full1080_ms=timeit(lambda: fn(big), 10)[0]) for n, fn in TECH.items()}
    # sensitivity sweeps on the hard (night+glare) images
    hard = [(t, im, m) for t, c, im, m in items if c != "day"]
    gh = [(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), m) for t, im, m in hard]
    blk = {b: float(np.mean([f1_mask(prep_B(g, b, 10), m) for g, m in gh]) * 100) for b in (11, 21, 31, 51, 101, 201)}
    Cs = {c: float(np.mean([f1_mask(prep_B(g, 31, c), m) for g, m in gh]) * 100) for c in (0, 5, 10, 20, 30)}
    def bg_noise(g, m): e = (m == 0); e = cv2.erode(e.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool); e[:14] = e[-14:] = False; return float(g[e].std())
    clip = {}
    for cl in (1, 2, 4, 8, 16, 40):
        f1s, amp = [], []
        for g, m in gh:
            e = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(g, (5, 5), 0))
            f1s.append(f1_mask(cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], m))
            amp.append(bg_noise(e, m) / max(bg_noise(g, m), 1e-3))
        clip[cl] = dict(f1=float(np.mean(f1s) * 100), noise_gain=float(np.mean(amp)))
    tiles = {t: float(np.mean([f1_mask(prep_C(g, 2.0, t), m) for g, m in gh]) * 100) for t in (2, 4, 8, 16)}
    order = dict(blur_then_clahe=float(np.mean([f1_mask(prep_C(g, 2.0, 8, True), m) for g, m in gh]) * 100),
                 clahe_then_blur=float(np.mean([f1_mask(prep_C(g, 2.0, 8, False), m) for g, m in gh]) * 100),
                 no_blur=float(np.mean([f1_mask(cv2.threshold(cv2.createCLAHE(2.0, (8, 8)).apply(g), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], m) for g, m in gh]) * 100))
    # annotation on a COPY vs on the original (evidence preservation)
    t, c, im, m = items[9]; g0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); base = char_acc(ocr(prep_C(g0)), t)
    ann = im.copy(); cv2.rectangle(ann, (6, 6), (514, 114), (0, 255, 0), 3); cv2.putText(ann, "TN09AB1234 0.91", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.polylines(ann, [np.array([[0, 100], [520, 20]])], False, (255, 0, 0), 3)
    in_place = im.copy(); cv2.rectangle(in_place, (6, 6), (514, 114), (0, 255, 0), 3); cv2.putText(in_place, "TN09AB1234 0.91", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.polylines(in_place, [np.array([[0, 100], [520, 20]])], False, (255, 0, 0), 3)
    annot = dict(plate=t, ocr_acc_clean=base * 100, ocr_acc_after_in_place=char_acc(ocr(prep_C(cv2.cvtColor(in_place, cv2.COLOR_BGR2GRAY))), t) * 100,
                 original_unchanged_when_copy_used=bool(np.array_equal(im, items[9][2])), original_unchanged_when_in_place=bool(np.array_equal(im, in_place)))
    save_json("t2.json", dict(rows=rows, summary=summ, timing=timing, block=blk, C=Cs, clip=clip, tiles=tiles, order=order, annot=annot))
    # ---- figures: side-by-side montage for day / night / glare + annotated corridor ----
    sel = [items[0], items[4], items[9]]; cells = []
    for t, c, im, m in sel:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); row = [cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)] + [cv2.cvtColor(fn(g), cv2.COLOR_GRAY2BGR) for fn in TECH.values()]
        for k, r in enumerate(row): cv2.putText(r, (["input"] + list(TECH))[k] + " | " + c, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if k else (0, 200, 0), 2)
        cells.append(np.hstack(row))
    cv2.imwrite(os.path.join(FIGS, "t2_montage.png"), np.vstack(cells))
    cv2.imwrite(os.path.join(FIGS, "t2_annotated.png"), np.hstack([im, ann, in_place]))
    # corridor frames (synthetic): lights on / lights off / tree-light flicker, with tripwire annotation drawn on a copy
    def corridor(level, flicker=0):
        H, W = 540, 960; f = np.zeros((H, W, 3), np.float32); yy, xx = np.mgrid[0:H, 0:W]
        f[:] = (0.25 + 0.5 * (yy / H))[..., None]; cv2.fillPoly(f, [np.array([[0, 0], [300, 160], [300, 380], [0, H]])], (0.2, 0.2, 0.25)); cv2.fillPoly(f, [np.array([[W, 0], [660, 160], [660, 380], [W, H]])], (0.2, 0.2, 0.25))
        cv2.rectangle(f, (300, 160), (660, 380), (0.6, 0.6, 0.65), -1)
        cv2.ellipse(f, (480, 330), (30, 70), 0, 0, 360, (0.12, 0.1, 0.1), -1)  # intruder silhouette
        f = f * level + flicker * np.exp(-(((xx - 700) ** 2) / (2 * 150 ** 2)))[..., None]
        return np.clip(f * 255 + rng.normal(0, 4, f.shape), 0, 255).astype(np.uint8)
    cor = [corridor(1.0), corridor(0.18), corridor(0.55, 0.5)]; crow = []
    for k, fr in enumerate(cor):
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); o = fr.copy(); cv2.polylines(o, [np.array([[250, 450], [700, 450]])], False, (0, 255, 255), 3); cv2.putText(o, "tripwire", (260, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        panels = [o] + [cv2.cvtColor(fn(g), cv2.COLOR_GRAY2BGR) for fn in TECH.values()]
        crow.append(np.hstack([cv2.resize(p, (320, 180)) for p in panels]))
    cv2.imwrite(os.path.join(FIGS, "t2_corridor.png"), np.vstack(crow))
    print(json.dumps(summ, indent=0)[:1500]); print(timing); print(blk, Cs); print(clip); print(tiles, order); print(annot)
