"""TASK 1 (installation + still images): cv2.imread / cv2.imdecode and BGR vs RGB."""
import cv2, numpy as np, os, sys
from common import *
p = os.path.join(RES, "red_patch.png"); img = np.zeros((40, 60, 3), np.uint8); img[:] = (0, 0, 255)           # pure RED written in BGR order
cv2.imwrite(p, img)
a = cv2.imread(p); b = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR); g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); miss = cv2.imread("does_not_exist.png")
rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB); plate = cv2.imread(os.path.join(DATA, "scenes", "frontal_00.jpg"))
out = dict(opencv=cv2.__version__, imread_shape=list(a.shape), imread_pixel_BGR=a[0, 0].tolist(), imdecode_equal_imread=bool(np.array_equal(a, b)), gray_shape=list(g.shape), gray_value=int(g[0, 0]),
           rgb_pixel_after_cvtColor=rgb[0, 0].tolist(), missing_file_returns=str(miss), scene_shape=list(plate.shape), python=sys.version.split()[0], contrib_face=hasattr(cv2, "face"))
save_json("t1_img.json", out); print(out)
