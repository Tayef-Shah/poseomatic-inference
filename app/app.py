from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pred.pose_estimator import preprocess_img, estimate_pose

app = FastAPI(title="Inference API")


class Img(BaseModel):
    img_url: str


@app.post("/estimate", status_code=200)
async def estimate(request: Img):
    img_tensor = preprocess_img(request.img_url)
    if img_tensor is None:
        raise HTTPException(status_code=404, detail="Image could not be downloaded")
    estimation = estimate_pose(img_tensor)
    # perhaps we upload image to cloud somewhere?
    return {"keypoints": estimation.tolist(), "status_code": 200}
