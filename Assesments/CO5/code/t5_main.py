"""TASK 5 - Expression classifiers: A HOG+SVM (cv2.ml), B small CNN (from scratch), C pre-trained EfficientNet-B0 (ONNX, transfer)."""
import cv2, numpy as np, os, time, json
import onnxruntime as ort
from common import *
d = np.load(os.path.join(DATA, "t5.npz")); X, y, tr, te, Xc = d["X"], d["y"], d["tr"], d["te"], d["Xc"]; yte = y[te]
CL = ["happy", "neutral", "unhappy"]
hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
def feat(g): return hog.compute(cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)).ravel().astype(np.float32)
Ftr = np.array([feat(g) for g in X[tr]]); ytr = y[tr]; str_ = d["subj"][tr]; t0 = time.perf_counter()
def balance(F, yy):                                   # replicate minority classes so that the SVM sees balanced classes
    idx = np.concatenate([np.resize(np.where(yy == c)[0], np.bincount(yy).max()) for c in range(3)]); return F[idx], yy[idx]
def fit_svm(F, yy, C, gam):
    Fb, yb = balance(F, yy); m = cv2.ml.SVM_create(); m.setType(cv2.ml.SVM_C_SVC); m.setKernel(cv2.ml.SVM_RBF); m.setC(C); m.setGamma(gam); m.train(Fb, cv2.ml.ROW_SAMPLE, yb.astype(np.int32)); return m
def cv_score(C, gam):                                 # leave-subjects-out CV inside the training subjects only
    subs = np.unique(str_); sc = []
    for k in range(0, len(subs), 2):
        v = np.isin(str_, subs[k:k + 2]); m = fit_svm(Ftr[~v], ytr[~v], C, gam); p = np.array([int(m.predict(f[None])[1][0, 0]) for f in Ftr[v]])
        mat = np.zeros((3, 3)); [mat.__setitem__((a, b), mat[a, b] + 1) for a, b in zip(ytr[v], p)]; rec = mat.diagonal() / mat.sum(1).clip(1); sc.append(rec.mean())
    return float(np.mean(sc))
grid = {(C, g): cv_score(C, g) for C in (1, 10, 100) for g in (0.0005, 0.002, 0.008)}; bestk = max(grid, key=grid.get)
svm = fit_svm(Ftr, ytr, *bestk); trainA = time.perf_counter() - t0; grid_out = {f"C={k[0]},gamma={k[1]}": v for k, v in grid.items()}
def predA(imgs): return np.array([int(svm.predict(feat(g)[None])[1][0, 0]) for g in imgs])
# C: pre-trained model (AffectNet 8 classes) mapped to the 3 kiosk classes
sess = ort.InferenceSession(os.path.join(MODELS, "emotion_enetb0.onnx"), providers=["CPUExecutionProvider"]); iname = sess.get_inputs()[0].name
EMO = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
def predC_one(g):
    im = cv2.cvtColor(cv2.resize(g, (224, 224)), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255; im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    o = sess.run(None, {iname: im.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]; k = EMO[int(o.argmax())]
    return 0 if k == "Happiness" else 1 if k == "Neutral" else 2
def predC(imgs): return np.array([predC_one(g) for g in imgs])
cnn = json.load(open(os.path.join(RES, "t5_cnn.json")))
def conf(yt, yp): m = np.zeros((3, 3), int); [m.__setitem__((a, b), m[a, b] + 1) for a, b in zip(yt, yp)]; return m
def stats(yt, yp):
    m = conf(yt, yp); rec = m.diagonal() / m.sum(1).clip(1); pre = m.diagonal() / m.sum(0).clip(1); f1 = 2 * rec * pre / (rec + pre).clip(1e-9)
    return dict(acc=float(m.diagonal().sum() / m.sum() * 100), macro_f1=float(f1.mean() * 100), recall=(rec * 100).tolist(), conf=m.tolist())
out = {}
pA, pAc = predA(X[te]), predA(Xc); pC, pCc = predC(X[te]), predC(Xc)
tA = timeit(lambda: predA(X[te][:1]), 30)[0]; tC = timeit(lambda: predC(X[te][:1]), 10)[0]
cfg = {"A HOG+SVM": (pA, pAc, tA, trainA, "-"), "B Small CNN": (np.array(cnn["pred"]), np.array(cnn["pred_candid"]), cnn["inf_ms"], cnn["train_s"], cnn["params"]),
       "C Pre-trained (ONNX)": (pC, pCc, tC, 0.0, "-")}
for k, (p, pc, t, tt, pr) in cfg.items(): out[k] = dict(posed=stats(yte, p), candid_proxy=stats(yte, pc), infer_ms=float(t), train_s=float(tt), params=pr)
out["svm_grid_cv_balanced_recall"] = grid_out; out["svm_best"] = list(bestk); out["majority_baseline_acc"] = float(np.bincount(yte).max() / len(yte) * 100); out["n_test"] = int(len(yte)); out["class_counts_test"] = np.bincount(yte).tolist(); out["class_counts_train"] = np.bincount(y[tr]).tolist()
save_json("t5.json", out)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(11, 6.5))
for j, (k, v) in enumerate(out.items()):
    if not isinstance(v, dict) or "posed" not in v: continue
    for i, key in enumerate(("posed", "candid_proxy")):
        m = np.array(v[key]["conf"]); a = ax[i][j]; a.imshow(m, cmap="Blues")
        for r in range(3):
            for c in range(3): a.text(c, r, m[r, c], ha="center", va="center", color="white" if m[r, c] > m.max() / 2 else "black")
        a.set_xticks(range(3)); a.set_xticklabels(CL, fontsize=8); a.set_yticks(range(3)); a.set_yticklabels(CL, fontsize=8); a.set_title(f"{k}\n{key} acc {v[key]['acc']:.0f}%", fontsize=9)
        a.set_xlabel("predicted", fontsize=8); a.set_ylabel("true", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "t5_confusion.png"), dpi=120)
for k, v in out.items():
    if isinstance(v, dict) and "posed" in v: print(k, "posed", round(v["posed"]["acc"], 1), "F1", round(v["posed"]["macro_f1"], 1), "rec", [round(x) for x in v["posed"]["recall"]], "| candid", round(v["candid_proxy"]["acc"], 1), "F1", round(v["candid_proxy"]["macro_f1"], 1), "| ms", round(v["infer_ms"], 1))
print("baseline", out["majority_baseline_acc"])
