"""Export the S-Learner (LightGBM) to ONNX so the web app can score users in the browser.

The S-Learner is one classifier on [f0..f11, treatment]. The app runs it twice per
user (treatment = 1 and treatment = 0) and takes the difference as the uplift.

    python -m src.export_onnx
"""
import sys

import joblib
import numpy as np
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from src.data import FEATURES, PROJECT_ROOT, load_criteo

OUT = PROJECT_ROOT / "app" / "public" / "model" / "s_learner.onnx"


def main():
    meta = joblib.load(PROJECT_ROOT / "models" / "best_uplift_model.joblib")
    clf = meta["model"].model                      # LGBMClassifier inside SLearner
    n_inputs = len(FEATURES) + 1
    onx = convert_lightgbm(clf, initial_types=[("input", FloatTensorType([None, n_inputs]))],
                           zipmap=False, target_opset=15)
    # Declare a dynamic batch size on every output so runtimes do not warn about shape {1} vs {N}.
    for out in onx.graph.output:
        dims = out.type.tensor_type.shape.dim
        if dims:
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "N"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(onx.SerializeToString())

    # Parity check against the Python model on 2,000 real users
    df = load_criteo().sample(2000, random_state=0)
    X = df[FEATURES].astype("float32")
    sess = ort.InferenceSession(str(OUT))
    name = sess.get_inputs()[0].name
    prob_out = [o.name for o in sess.get_outputs() if "prob" in o.name.lower()][0]

    def onnx_p(t):
        arr = np.hstack([X.values, np.full((len(X), 1), t, dtype="float32")])
        return sess.run([prob_out], {name: arr})[0][:, 1]

    uplift_onnx = onnx_p(1) - onnx_p(0)
    uplift_py = meta["model"].predict_uplift(X)
    err = np.abs(uplift_onnx - uplift_py).max()
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) · outputs: {[o.name for o in sess.get_outputs()]}")
    print(f"Max |uplift difference| ONNX vs Python on 2,000 users: {err:.2e}")
    if err > 1e-4:
        sys.exit("Parity check failed")


if __name__ == "__main__":
    main()
