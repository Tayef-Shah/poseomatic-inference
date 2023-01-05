from pred.models.tf_pred import *
from typing import Any


def tf_run_classifier(image: str) -> Any:
    img = load_image(image)
    if img is None:
        return None
    pred_results = tf_predict(img)
    pred_results["status_code"] = 200
    return pred_results
