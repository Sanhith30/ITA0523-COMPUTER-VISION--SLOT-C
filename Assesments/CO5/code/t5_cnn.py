"""Technique B: small CNN trained from scratch (TensorFlow-CPU, separate venv). Writes predictions for the shared test sets."""
import numpy as np, time, os, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(1); np.random.seed(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); d = np.load(os.path.join(ROOT, "data", "t5.npz"))
def prep(a): return tf.image.resize(a[..., None].astype(np.float32), (48, 48), antialias=True).numpy() / 255.0
Xtr, ytr = prep(d["X"][d["tr"]]), d["y"][d["tr"]]; Xte, yte = prep(d["X"][d["te"]]), d["y"][d["te"]]; Xc = prep(d["Xc"])
model = keras.Sequential([layers.Input((48, 48, 1)), layers.RandomFlip("horizontal"), layers.RandomRotation(.06), layers.RandomTranslation(.08, .08), layers.RandomContrast(.3),
    layers.Conv2D(16, 3, activation="relu", padding="same"), layers.MaxPooling2D(), layers.Conv2D(32, 3, activation="relu", padding="same"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"), layers.MaxPooling2D(), layers.GlobalAveragePooling2D(), layers.Dense(48, activation="relu"), layers.Dropout(.4), layers.Dense(3, activation="softmax")])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
cw = {i: len(ytr) / (3 * c) for i, c in enumerate(np.bincount(ytr))}
t0 = time.time(); model.fit(Xtr, ytr, epochs=80, batch_size=16, class_weight=cw, verbose=0); train_s = time.time() - t0
p = model.predict(Xte, verbose=0).argmax(1); pc = model.predict(Xc, verbose=0).argmax(1)
one = Xte[:1]; model.predict(one, verbose=0); t0 = time.time(); [model.predict(one, verbose=0) for _ in range(30)]; inf_ms = (time.time() - t0) / 30 * 1000
json.dump(dict(pred=p.tolist(), pred_candid=pc.tolist(), train_s=train_s, inf_ms=inf_ms, params=int(model.count_params()), train_acc=float(model.evaluate(Xtr, ytr, verbose=0)[1])), open(os.path.join(ROOT, "results", "t5_cnn.json"), "w"))
print("cnn acc", (p == yte).mean(), "candid", (pc == yte).mean(), "train_s", train_s, "inf_ms", inf_ms)
