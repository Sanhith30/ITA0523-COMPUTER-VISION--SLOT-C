"""TASK 7 - Corridor intrusion: A frame differencing | B MOG2 | C Farneback optical flow. Synthetic 25 fps night-corridor clips with ground truth."""
import cv2, numpy as np, time
from common import *
W, H, FPS, NF = 320, 180, 25, 1500
ROI = np.array([[40, 60], [260, 60], [260, 170], [40, 170]], np.int32)
yy, xx = np.mgrid[0:H, 0:W]; rng0 = np.random.default_rng(1)
BASE = (110 + 45 * yy / H + cv2.GaussianBlur(rng0.normal(0, 14, (H, W)).astype(np.float32), (0, 0), 6) * 2).astype(np.float32)
def render(scn, seed=0):
    rng = np.random.default_rng(seed); frames = []; enter = None; xs = []
    v = {"fast": 3.0, "slow": 0.4}.get(scn.get("intr"), 0); t0 = 250
    for t in range(NF):
        f = BASE.copy(); gain = 1.0
        for (ts, g) in scn.get("light", []):
            if t >= ts: gain = g
        if scn.get("ramp") and 600 < t < 700: gain *= 1 + 0.8 * (t - 600) / 100
        f *= gain
        if scn.get("tree"):
            amp = 26; wind = 0.32 + 0.12 * np.sin(t / 90)
            for k in range(9):
                cx = 262 + amp * np.sin(wind * t + k * 1.7) * (0.5 + 0.5 * np.sin(k)) ; cy = 20 + 18 * k
                cv2.circle(f, (int(cx), int(cy)), 11, 35 * gain, -1)
        if v:
            x = min(-10 + v * max(0, t - t0), 150) if t >= t0 else -30
            cv2.ellipse(f, (int(x), 120), (7, 20), 0, 0, 360, 30 * gain, -1)
            if enter is None and x >= 40: enter = t
        f = np.clip(f + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8); frames.append(f); xs.append(x if v and t >= t0 else None)
    return (frames, enter, xs)
SCN = {"S1 quiet": dict(), "S2 lights": dict(light=[(300, 1.9), (600, 1.0), (900, 0.5), (1200, 1.0)], ramp=True), "S3 tree": dict(tree=True),
       "S4 fast intruder + tree": dict(tree=True, intr="fast", light=[(1000, 1.6)]), "S5 slow intruder + tree": dict(tree=True, intr="slow")}
def blobs(mask, amin):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = []
    for c in cs:
        a = cv2.contourArea(c)
        if a >= amin:
            m = cv2.moments(c); cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]; out.append((cx, cy, a, cv2.boundingRect(c)))
    return out
class A:
    name = "A Frame differencing"
    def __init__(s): s.prev = None
    def step(s, g):
        b = cv2.GaussianBlur(g, (5, 5), 0)
        if s.prev is None: s.prev = b; return np.zeros_like(g)
        m = cv2.threshold(cv2.absdiff(b, s.prev), 25, 255, cv2.THRESH_BINARY)[1]; s.prev = b; return cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
class B:
    def __init__(s, lr=-1): s.m = cv2.createBackgroundSubtractorMOG2(300, 25, True); s.lr = lr; s.name = "B MOG2 lr=" + ("auto" if lr < 0 else str(lr))
    def step(s, g):
        m = s.m.apply(g, learningRate=s.lr); m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)[1]; return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
class C:
    name = "C Farneback flow"
    def __init__(s): s.prev = None
    def step(s, g):
        sm = cv2.resize(cv2.GaussianBlur(g, (5, 5), 0), (160, 90), interpolation=cv2.INTER_AREA)
        if s.prev is None: s.prev = sm; return np.zeros_like(g)
        fl = cv2.calcOpticalFlowFarneback(s.prev, sm, None, 0.5, 3, 15, 3, 5, 1.2, 0); s.prev = sm; mag = np.linalg.norm(fl, axis=2)
        m = cv2.morphologyEx((mag > 0.7).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)); return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
def run(method, frames, enter, amin=60, need=5, refr=75, guard=False, record=False, xs=None):
    alerts, ms_t, cnt, last, det_hist, vis = [], [], 0, -999, [], []; cnti, deti, first_i = 0, [], None
    for t, g in enumerate(frames):
        t0 = time.perf_counter(); m = method.step(g); bl = blobs(m, amin); ms_t.append(ms(t0))
        fg = (m > 0).mean(); inroi = [b for b in bl if cv2.pointPolygonTest(ROI, (b[0], b[1]), False) >= 0]
        det = bool(inroi) and not (guard and fg > 0.4); det_hist.append(det); cnt = cnt + 1 if det else 0
        di = bool(xs and xs[t] is not None and any(b[3][0] < xs[t] + 10 and b[3][0] + b[3][2] > xs[t] - 10 and b[3][1] < 140 and b[3][1] + b[3][3] > 100 and b[0] > 30 for b in bl)) and not (guard and fg > 0.4); deti.append(di); cnti = cnti + 1 if di else 0
        if cnti >= need and first_i is None: first_i = t
        if cnt >= need and t - last > refr: alerts.append(t); last = t
        if record: vis.append((m, bl, alerts[-1:] and t - alerts[-1] < 40))
    ev = dict(alerts=alerts, ms=float(np.mean(ms_t)), det=det_hist)
    if enter is not None:
        ev["delay_s"] = (max(first_i, enter) - enter) / FPS if first_i is not None else None
        ev["false_before"] = len([a for a in alerts if a < enter - 5]); ev["persist_pct"] = float(np.mean(deti[-250:]) * 100)
    else: ev["false_alarms"] = len(alerts)
    return ev, vis
if __name__ == "__main__":
    raw = {k: render(v, seed=i) for i, (k, v) in enumerate(SCN.items())}; data = {k: (a, b) for k, (a, b, c) in raw.items()}; XS = {k: c for k, (a, b, c) in raw.items()}; out = {}
    meths = lambda: [A(), B(-1), C()]
    for guard in (False, True):
        for k, (fr, en) in data.items():
            for m in meths():
                ev, _ = run(m, fr, en, guard=guard, xs=XS[k]); out[f"{m.name}|{k}|guard={guard}"] = {a: b for a, b in ev.items() if a != "det"}
    # learning-rate study for MOG2 (guard on): slow intruder detection and false alarms on no-intruder clips
    lrs = {}
    for lr in (0.0, 0.0005, 0.005, 0.05):
        r = {}
        for k in ("S2 lights", "S3 tree", "S5 slow intruder + tree"):
            ev, _ = run(B(lr), data[k][0], data[k][1], guard=True, xs=XS[k]); r[k] = {a: b for a, b in ev.items() if a != "det"}
        ev, _ = run(B(lr), data["S2 lights"][0], None, guard=False); r["S2 lights (no global-change guard)"] = {a: b for a, b in ev.items() if a != "det"}
        lrs[lr] = r
    # speed at the processing size, per-frame (ms) and fps on this machine
    fr = data["S5 slow intruder + tree"][0]; speed = {}
    for m in meths(): t = [ms_ for ms_ in [run(m, fr[:300], None)[0]["ms"]]]; speed[m.name] = t[0]
    # annotated output videos (slow intruder) - 3 methods side by side
    res = {}; vids = {}
    for m in meths(): ev, vis = run(m, fr, data["S5 slow intruder + tree"][1], guard=True, record=True, xs=XS["S5 slow intruder + tree"]); res[m.name] = (ev, vis)
    vw = cv2.VideoWriter(os.path.join(VID, "t7_corridor_slow_intruder.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W * 3, H)); keyframes = {}
    for t in range(NF):
        row = []
        for n, (ev, vis) in res.items():
            f = cv2.cvtColor(fr[t], cv2.COLOR_GRAY2BGR); cv2.polylines(f, [ROI], True, (0, 255, 255), 1)
            for (cx, cy, a, (x, y, w, h)) in vis[t][1]: cv2.rectangle(f, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if vis[t][2]: cv2.putText(f, "ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
            cv2.putText(f, n[:18], (5, H - 6), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1); row.append(f)
        img = np.hstack(row); vw.write(img)
        if t in (200, 420, 700): keyframes[t] = img
    vw.release(); cv2.imwrite(os.path.join(FIGS, "t7_frames.png"), np.vstack(list(keyframes.values())))
    save_json("t7.json", dict(res=out, lr=lrs, speed_ms=speed, enter={k: v[1] for k, v in data.items()}, minutes=NF / FPS / 60 * 3, fps=FPS, n=NF))
    for k, v in out.items():
        if "guard=True" in k: print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a != "alerts"})
    print(lrs); print(speed)
