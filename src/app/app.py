from fastapi import FastAPI, HTTPException
from pred.image_classifier import *
from pydantic import BaseModel

app = FastAPI(title="Inference API")


class Img(BaseModel):
    img_url: str


@app.post("/predict", status_code=200)
async def predict_tf(request: Img):
    prediction = tf_run_classifier(request.img_url)
    if not prediction:
        raise HTTPException(status_code=404, detail="Image could not be downloaded")
    return prediction
