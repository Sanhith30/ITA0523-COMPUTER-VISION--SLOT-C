"""Create a synthetic 1080p, 25 fps 'gate camera' clip (MJPG/AVI: every frame is intra-coded so seeking is cheap)."""
import cv2, numpy as np, os
from common import VID
W, H, N, FPS = 1920, 1080, 125, 25
path = os.path.join(VID, "gate_1080p.avi")
vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
rng = np.random.default_rng(0)
yy, xx = np.mgrid[0:H, 0:W]
base = np.stack([(xx * 255 // W), (yy * 255 // H), ((xx + yy) * 255 // (W + H))], -1).astype(np.uint8)
tex = (rng.random((H // 8, W // 8, 3)) * 60).astype(np.uint8)
tex = cv2.resize(tex, (W, H), interpolation=cv2.INTER_NEAREST)
for i in range(N):
    f = cv2.add(base // 2, tex)
    x = int((i * 37) % (W - 300)); cv2.rectangle(f, (x, 400), (x + 260, 900), (30, 200, 240), -1)
    cv2.circle(f, (W - x - 150, 300), 90, (240, 240, 240), -1)
    cv2.putText(f, f"frame {i:03d}", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    vw.write(f)
vw.release(); print(path, os.path.getsize(path) / 1e6, "MB")
