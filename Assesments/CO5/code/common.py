"""Common helpers for the Assignment-5 experiments (OpenCV 4.10 CPU-only)."""
import os, json, time
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, MODELS, RES, FIGS, VID = (os.path.join(ROOT, d) for d in ("data", "models", "results", "figs", "videos"))
for d in (RES, FIGS, VID): os.makedirs(d, exist_ok=True)
cv2.setNumThreads(1)           # the sandbox has one core; keep OpenCV single-threaded for fair timing
def save_json(name, obj):
    with open(os.path.join(RES, name), "w") as f: json.dump(obj, f, indent=1, default=float)
def load_json(name):
    with open(os.path.join(RES, name)) as f: return json.load(f)
def ms(t0): return (time.perf_counter() - t0) * 1000.0
def timeit(fn, n=10, warm=2):
    for _ in range(warm): fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(ms(t0))
    return float(np.mean(t)), float(np.std(t))
