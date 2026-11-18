"""TASK 6 - Plate OCR: localise + perspective-correct (cv2.getPerspectiveTransform), then A Tesseract | B contour segmentation + kNN | C deep OCR (PP-OCR CRNN via RapidOCR/ONNX)."""
import cv2, numpy as np, re, time, collections, pytesseract
from rapidocr_onnxruntime import RapidOCR
from common import *
rng = np.random.default_rng(31)
F_TRAIN = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX]; F_NEW = cv2.FONT_HERSHEY_TRIPLEX
CH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; PW, PH = 520, 120
STATES = ["TN", "KA", "MH", "DL", "AP", "KL", "TS", "GJ", "UP", "WB", "RJ", "HR"]
def rand_plate():
    return rng.choice(STATES) + f"{rng.integers(1, 99):02d}" + "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), 2)) + f"{rng.integers(0, 10000):04d}"
def render_plate(text, font):
    img = np.full((PH, PW, 3), 255, np.uint8); cv2.rectangle(img, (4, 4), (PW - 5, PH - 5), (0, 0, 0), 3)
    sc = 2.3 if font != F_NEW else 2.0; th = 5 if font != F_NEW else 4
    (tw, thh), _ = cv2.getTextSize(text, font, sc, th); cv2.putText(img, text, ((PW - tw) // 2, (PH + thh) // 2), font, sc, (0, 0, 0), th, cv2.LINE_AA); return img
def make_scene(text, cond):
    plate = render_plate(text, F_NEW if cond == "new font" else cv2.FONT_HERSHEY_DUPLEX)
    if cond == "dirt":
        for _ in range(14): cv2.circle(plate, (int(rng.integers(10, PW - 10)), int(rng.integers(10, PH - 10))), int(rng.integers(4, 11)), (int(rng.integers(40, 120)),) * 3, -1)
        plate = np.clip(plate.astype(np.int16) + rng.normal(0, 18, plate.shape), 0, 255).astype(np.uint8)
    H, W = 450, 800; scene = np.clip(np.full((H, W, 3), 85, np.float32) + rng.normal(0, 8, (H, W, 3)), 0, 255).astype(np.uint8); cv2.rectangle(scene, (120, 120), (680, 330), (45, 45, 55), -1)
    k = {"skew": .30}.get(cond, .06); cx, cy, w, h = 400, 225, 440, 106
    dst = np.float32([[cx - w / 2 + rng.uniform(-1, 1) * k * 90, cy - h / 2 + rng.uniform(-1, 1) * k * 70], [cx + w / 2 + rng.uniform(-1, 1) * k * 90, cy - h / 2 + rng.uniform(-1, 1) * k * 70],
                      [cx + w / 2 + rng.uniform(-1, 1) * k * 90, cy + h / 2 + rng.uniform(-1, 1) * k * 70], [cx - w / 2 + rng.uniform(-1, 1) * k * 90, cy + h / 2 + rng.uniform(-1, 1) * k * 70]])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]]), dst); wp = cv2.warpPerspective(plate, M, (W, H)); mk = cv2.warpPerspective(np.full((PH, PW), 255, np.uint8), M, (W, H)) > 0
    scene[mk] = wp[mk]
    if cond == "blur":
        ker = np.zeros((15, 15), np.float32); ker[7, :] = 1 / 15; scene = cv2.GaussianBlur(cv2.filter2D(scene, -1, ker), (0, 0), 1.8)
    if cond == "glare":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); cxg = rng.uniform(300, 500); veil = 0.9 * np.exp(-(((xx - cxg) ** 2 + (yy - 225) ** 2) / (2 * 75 ** 2)))
        scene = np.clip(scene.astype(np.float32) * .85 + 255 * veil[..., None], 0, 255).astype(np.uint8)
    return np.clip(scene.astype(np.float32) + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
def order(pts):
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel(); return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])
def localise(scene):
    g = cv2.GaussianBlur(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), (5, 5), 0); _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)); cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); best = None
    for c in cs:
        a = cv2.contourArea(c)
        if a < 0.04 * g.size or a > 0.6 * g.size: continue
        ap = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True); x, y, w, h = cv2.boundingRect(c)
        if len(ap) == 4 and 2.0 < w / h < 7.0 and (best is None or a > best[0]): best = (a, ap.reshape(4, 2).astype(np.float32))
    return None if best is None else order(best[1])
def rectify(scene, corners):
    M = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])); return cv2.warpPerspective(scene, M, (PW, PH))
def binarise(g): g = cv2.GaussianBlur(g, (5, 5), 0); e = cv2.createCLAHE(2.0, (8, 8)).apply(g); return cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# ---- A Tesseract ----
def ocr_A(rect):
    b = binarise(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)); im = cv2.copyMakeBorder(b, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    d = pytesseract.image_to_data(im, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", output_type=pytesseract.Output.DICT)
    txt = "".join(t for t in d["text"] if t.strip()); confs = [float(c) for c, t in zip(d["conf"], d["text"]) if t.strip() and float(c) >= 0]
    return re.sub(r"[^A-Z0-9]", "", txt), (float(np.mean(confs)) if confs else 0.0)
# ---- B contour segmentation + kNN ----
def glyph(img_bin):                                    # black glyph on white -> 20x20 normalised vector
    ys, xs = np.where(img_bin == 0)
    if len(ys) == 0: return None
    c = img_bin[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; s = max(c.shape); sq = np.full((s, s), 255, np.uint8); sq[(s - c.shape[0]) // 2:(s - c.shape[0]) // 2 + c.shape[0], (s - c.shape[1]) // 2:(s - c.shape[1]) // 2 + c.shape[1]] = c
    return (cv2.resize(sq, (20, 20), interpolation=cv2.INTER_AREA).astype(np.float32) / 255).ravel()
def train_knn():
    X, y = [], []
    for ch in CH:
        for f in F_TRAIN:
            for sc in (1.8, 2.3, 3.0):
                for th in (3, 5, 7):
                    for rot in (-6, 0, 6):
                        im = np.full((140, 140), 255, np.uint8); cv2.putText(im, ch, (35, 100), f, sc, 0, th, cv2.LINE_AA)
                        M = cv2.getRotationMatrix2D((70, 70), rot, 1); im = cv2.warpAffine(im, M, (140, 140), borderValue=255)
                        for noise in (0, 1):
                            t = im if not noise else cv2.GaussianBlur(np.clip(im + rng.normal(0, 25, im.shape), 0, 255).astype(np.uint8), (3, 3), 0)
                            g = glyph(cv2.threshold(t, 127, 255, cv2.THRESH_BINARY)[1])
                            if g is not None: X.append(g); y.append(CH.index(ch))
    k = cv2.ml.KNearest_create(); k.train(np.array(X, np.float32), cv2.ml.ROW_SAMPLE, np.array(y, np.float32)); return k, len(X)
KNN, NTRAIN = train_knn()
def ocr_B(rect):
    b = binarise(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)); inv = 255 - b; cs, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); boxes = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if 0.35 * PH < h < 0.85 * PH and w < 0.16 * PW and cv2.contourArea(c) > 80: boxes.append((x, y, w, h))
    boxes.sort(); txt, dist = "", []
    for x, y, w, h in boxes:
        g = glyph(b[y:y + h, x:x + w])
        if g is None: continue
        _, r, nb, d = KNN.findNearest(g[None], 3); txt += CH[int(r[0, 0])]; dist.append(float(d.mean()))
    return txt, (100 - 10 * float(np.mean(dist)) if dist else 0.0)
# ---- C deep OCR ----
RAP = RapidOCR()
def ocr_C(rect):
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)); r, _ = RAP(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), use_det=False, use_cls=False, use_rec=True)
    return (re.sub(r"[^A-Z0-9]", "", r[0][0].upper()), float(r[0][1]) * 100) if r else ("", 0.0)
def lev_ops(a, b):
    n, m = len(a), len(b); D = np.zeros((n + 1, m + 1), int); D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1): D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, subs = n, m, []
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]: subs.append((a[i - 1], b[j - 1]))
            i -= 1; j -= 1
        elif D[i, j] == D[i - 1, j] + 1: i -= 1
        else: j -= 1
    return int(D[n, m]), subs
PAT = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")
if __name__ == "__main__":
    CONDS = ["clean", "blur", "skew", "glare", "dirt", "new font"]; N = 10; rows = []; examples = {}
    for cond in CONDS:
        for i in range(N):
            gt = rand_plate(); sc = make_scene(gt, cond); cn = localise(sc)
            if cond in ("clean", "skew", "glare", "blur") and i == 0: examples[cond] = (sc.copy(), cn)
            if cn is None: rect = None
            else: rect = rectify(sc, cn)
            for name, fn in (("A Tesseract", ocr_A), ("B Contour+kNN", ocr_B), ("C Deep OCR", ocr_C)):
                if rect is None: txt, conf, dt = "", 0.0, 0.0
                else: t = time.perf_counter(); txt, conf = fn(rect); dt = ms(t)
                e, subs = lev_ops(txt, gt); rows.append(dict(cond=cond, tech=name, gt=gt, pred=txt, conf=conf, ms=dt, char_acc=max(0, 1 - e / len(gt)), exact=txt == gt, subs=subs, localised=cn is not None))
    summ = {}
    for n in ("A Tesseract", "B Contour+kNN", "C Deep OCR"):
        for c in CONDS + ["all"]:
            r = [x for x in rows if x["tech"] == n and (c == "all" or x["cond"] == c)]
            summ[f"{n}|{c}"] = dict(char_acc=float(np.mean([x["char_acc"] for x in r]) * 100), plate_acc=float(np.mean([x["exact"] for x in r]) * 100), ms=float(np.mean([x["ms"] for x in r if x["localised"]] or [0])))
    errs = {n: collections.Counter(f"{a}->{b}" for x in rows if x["tech"] == n for a, b in x["subs"]).most_common(8) for n in ("A Tesseract", "B Contour+kNN", "C Deep OCR")}
    loc = {c: float(np.mean([x["localised"] for x in rows if x["cond"] == c and x["tech"] == "A Tesseract"]) * 100) for c in CONDS}
    # what-if: Tesseract on the raw scene (no localisation) and on an un-deskewed axis-aligned crop
    raw, crop = [], []
    for cond in ("clean", "skew"):
        for i in range(10):
            gt = rand_plate(); sc = make_scene(gt, cond); g = cv2.cvtColor(sc, cv2.COLOR_BGR2GRAY)
            t = re.sub(r"[^A-Z0-9]", "", pytesseract.image_to_string(g, config="--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")); raw.append((cond, max(0, 1 - lev_ops(t, gt)[0] / len(gt)), t == gt))
            cn = localise(sc)
            if cn is not None:
                x, y, w, h = cv2.boundingRect(cn.astype(np.int32)); t2, _ = ocr_A(cv2.resize(sc[y:y + h, x:x + w], (PW, PH))); crop.append((cond, max(0, 1 - lev_ops(t2, gt)[0] / len(gt)), t2 == gt))
    whatif = {k: {c: dict(char_acc=float(np.mean([a for cc, a, e in v if cc == c]) * 100 if any(cc == c for cc, a, e in v) else -1), plate_acc=float(np.mean([e for cc, a, e in v if cc == c]) * 100 if any(cc == c for cc, a, e in v) else -1)) for c in ("clean", "skew")} for k, v in (("raw_no_localisation", raw), ("crop_no_deskew", crop))}
    # verification rule: accept only if regex valid AND confidence >= tau; measure wrong-accept rate
    ver = {}
    for n in ("A Tesseract", "B Contour+kNN", "C Deep OCR"):
        R = [x for x in rows if x["tech"] == n]; ver[n] = {}
        for tau in (0, 60, 80, 90, 95):
            acc = [x for x in R if PAT.match(x["pred"]) and x["conf"] >= tau]
            ver[n][tau] = dict(accepted_pct=len(acc) / len(R) * 100, wrong_among_accepted_pct=(float(np.mean([not x["exact"] for x in acc]) * 100) if acc else 0.0), accepted=len(acc))
    # ID-card lines: Tesseract vs deep OCR
    idc = {"A Tesseract": [], "C Deep OCR": []}
    for i in range(12):
        txt = "NAME ARUN KUMAR ID 1923%05d" % int(rng.integers(0, 99999)); im = np.full((90, 760, 3), 255, np.uint8); cv2.putText(im, txt, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        if i % 2: im = cv2.GaussianBlur(im, (0, 0), 1.3); im = np.clip(im + rng.normal(0, 12, im.shape), 0, 255).astype(np.uint8)
        t0 = time.perf_counter(); a = re.sub(r"[^A-Z0-9 ]", "", pytesseract.image_to_string(im, config="--psm 7")).strip().upper(); ta = ms(t0)
        t0 = time.perf_counter(); r, _ = RAP(im, use_det=False, use_cls=False, use_rec=True); c = r[0][0].upper() if r else ""; tc = ms(t0)
        for n, p, tt in (("A Tesseract", a, ta), ("C Deep OCR", c, tc)): idc[n].append(dict(char_acc=max(0, 1 - lev_ops(re.sub(r"[^A-Z0-9 ]", "", p), txt)[0] / len(txt)), ms=tt))
    idsum = {n: dict(char_acc=float(np.mean([x["char_acc"] for x in v]) * 100), ms=float(np.mean([x["ms"] for x in v]))) for n, v in idc.items()}
    save_json("t6.json", dict(summary=summ, errors=errs, localised_pct=loc, whatif=whatif, verification=ver, idcards=idsum, knn_train=NTRAIN, n_per_cond=N, rows=[{k: v for k, v in r.items() if k != "subs"} for r in rows]))
    ex = []
    for k, (sc, cn) in examples.items():
        a = sc.copy()
        if cn is not None: cv2.polylines(a, [cn.astype(np.int32)], True, (0, 255, 0), 3)
        cv2.putText(a, k, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3); ex.append(cv2.resize(a, (400, 225)))
    cv2.imwrite(os.path.join(FIGS, "t6_scenes.png"), np.vstack([np.hstack(ex[:2]), np.hstack(ex[2:4])]))
    for k, v in summ.items():
        if k.endswith("|all") or True: print(k, {a: round(b, 1) for a, b in v.items()})
    print(loc); print(errs); print(whatif); print(json.dumps(ver)[:700]); print(idsum)
