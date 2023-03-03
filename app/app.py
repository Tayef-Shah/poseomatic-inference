import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from pred.pose_estimator import preprocess_img, estimate_pose
from utils.pose_vis import draw_prediction_on_image
from utils.s3client import S3Client

app = FastAPI(title="Inference API")


class Img(BaseModel):
    img_url: str


s3_client = S3Client(region_name="ca-central-1", bucket_name="poseomatic")


@app.post("/estimate", status_code=200)
async def estimate(request: Img):
    request_image = s3_client.load_image(request.img_url)
    if request_image is None:
        raise HTTPException(status_code=404, detail="Image could not be downloaded")
    img_tensor = preprocess_img(request_image)
    estimation = estimate_pose(img_tensor)

    # draw estimation over image
    display_image = tf.expand_dims(request_image, axis=0)
    display_image = tf.cast(
        tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32
    )
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), estimation
    )

    img = Image.fromarray(output_overlay)
    s3_client.upload_image(img, request.img_url)

    return {"keypoints": estimation.tolist(), "status_code": 200}
