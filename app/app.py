import os
import logging
import cv2
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from pred.pose_estimator import (
    preprocess_img,
    preprocess_video,
    init_crop_region,
    determine_crop_region,
    run_inference,
    run_inference_no_crop,
)
from utils.pose_vis import draw_prediction_on_image
from utils.s3client import S3Client
import tensorflow_hub as hub
import skvideo.io
from moviepy.editor import ImageSequenceClip
from pred.models.movenet import movenet
from utils.videoio import VideoIO
from keypoints_from_video import main as make_lookup
from calculations import get_Score


# eh do sth here idk


app = FastAPI(title="Inference API")
logging.basicConfig(format="%(levelname)s:     %(message)s", level=logging.INFO)
s3_client = S3Client(region_name="ca-central-1", bucket_name="poseomatic")
logging.info("Started S3 client")
logging.info("Begining model download...")
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
logging.info("Model download complete")
os.mkdir("lookup")


class Img(BaseModel):
    img_url: str


class Video(BaseModel):
    video_url: str


class CmpRequest(BaseModel):
    reference_video: str
    reference_video_url: str
    user_video: str
    user_video_url: str


@app.post("/estimate", status_code=200)
async def estimate(request: Img):
    logging.info("Loading image from S3...")
    request_image = s3_client.load_image(request.img_url)
    if request_image is None:
        raise HTTPException(status_code=404, detail="Image could not be downloaded")
    img_tensor = preprocess_img(request_image)

    num_frames, image_height, image_width, _ = img_tensor.shape
    crop_region = init_crop_region(image_height, image_width)
    crop_size = [256, 256]

    frame_idx = 0

    logging.info("Running pose estimation...")
    estimation = run_inference(
        module, img_tensor[frame_idx, :, :, :], crop_region, crop_size
    )
    logging.info("Estimation complete")
    crop_region = determine_crop_region(estimation, image_height, image_width)

    logging.info("Drawing estimation landmarks onto image...")
    display_image = tf.expand_dims(request_image, axis=0)
    display_image = tf.cast(
        tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32
    )
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), estimation
    )

    img = Image.fromarray(output_overlay)
    logging.info("Uploading image to S3...")
    s3_client.upload_image(img, request.img_url)

    return {"file_name": "estimation_" + request.img_url, "status_code": 200}


@app.post("/v1/estimate-video", status_code=200)
async def estimate_video(request: Video):
    logging.info("Loading video from S3...")
    src_video = s3_client.load_video(request.video_url)
    if src_video is None:
        raise HTTPException(status_code=404, detail="Video could not be downloaded")

    original_frames, video_tensor = preprocess_video(src_video)
    display_frames = []
    for frame in original_frames:
        display_image = tf.expand_dims(frame, axis=0)
        display_image = tf.cast(
            tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32
        )
        display_frames.append(display_image)

    num_frames, frame_height, frame_width, _ = video_tensor.shape
    logging.info(f"Processed video with shape {video_tensor.shape}")

    crop_region = init_crop_region(frame_height, frame_width)
    crop_size = [256, 256]

    output_frames = []
    for frame_idx in range(num_frames):
        logging.info(f"Frame {frame_idx} : Running pose estimation...")
        estimation = run_inference(
            module, video_tensor[frame_idx, :, :, :], crop_region, crop_size
        )
        logging.info(f"Frame {frame_idx} : Estimation complete")
        logging.info(f"Frame {frame_idx} : Drawing estimation landmarks onto image...")
        output_overlay = draw_prediction_on_image(
            np.squeeze(display_frames[frame_idx].numpy(), axis=0), estimation
        )
        output_frames.append(output_overlay)
        crop_region = determine_crop_region(estimation, frame_height, frame_width)

    output_video = np.stack(output_frames, axis=0)

    # set up the video encoding parameters
    outputfile = "estimation_" + request.video_url
    videocodec = "libx264"
    fps = 30.0

    # encode the video using scikit-video
    skvideo.io.vwrite(
        outputfile,
        output_video,
        inputdict={"-r": str(fps)},
        outputdict={"-vcodec": videocodec},
    )

    s3_client.upload_video(outputfile)

    return {"file_name": "estimation_" + request.video_url}


@app.post("/v2/estimate-video", status_code=200)
async def estimate_video(request: Video):
    url = s3_client.s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": s3_client.bucket, "Key": request.video_url},
    )

    cap = cv2.VideoCapture(url)
    logging.info("Downloaded S3 video")
    frames = []
    success, frame_image = cap.read()
    count = 0
    while success:
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        frames.append(frame_image)
        success, frame_image = cap.read()
        count += 1

    logging.info(f"Total frames: {count}")
    video = np.stack(frames, axis=0)
    logging.info(f"Video shape: {video.shape}")
    cap.release()

    output_frames = []
    frame_idx = 0
    for f in frames:
        input_size = 256
        input_image = tf.expand_dims(f, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # Run model inference.
        keypoints_with_scores = movenet(module, input_image)

        # Visualize the predictions with image.
        logging.info(f"Frame {frame_idx} : Drawing estimation landmarks onto image...")
        display_image = tf.expand_dims(f, axis=0)
        display_image = tf.cast(
            tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32
        )
        output_overlay = draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores
        )
        output_frames.append(output_overlay)
        frame_idx += 1

    logging.info("Writing video file...")
    file_key = "estimation_" + request.video_url
    clip = ImageSequenceClip(output_frames, fps=30)
    clip.write_videofile(file_key, audio=False)

    logging.info("Uploading to S3...")
    s3_client.upload_video(file_key)

    return {"file_name": file_key}


@app.post("/compare", status_code=200)
async def compare_videos(request: CmpRequest):
    video_processor = VideoIO()
    result_file_name = ""
    ref_video_name = request.reference_video
    usr_video_name = request.user_video
    # eh do sth here idk
    ref_url = s3_client.s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": s3_client.bucket, "Key": request.reference_video},
    )
    usr_url = s3_client.s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": s3_client.bucket, "Key": request.user_video},
    )

    # 1. Video Processing

    logging.info(f"Processing video: {ref_video_name}")
    reference_frames = video_processor.split_into_frames(ref_url)
    logging.info(f"Proccessing video: {usr_video_name}")
    user_frames = video_processor.split_into_frames(usr_url)

    # 3. Pose Comparision
    lookup = os.path.join("lookup/", f"{ref_video_name}.pickle")
    if not os.path.exists(lookup):
        make_lookup(video=ref_url, activity=ref_video_name, lookup=lookup)
    g = get_Score(lookup)
    final_score, score_list = g.calculate_Score(usr_url, ref_video_name)
    print("Total Score : ", final_score)
    print("Score List : ", score_list)

    # 2. Pose Estimation

    output_frames_reference = []
    frame_idx = 0
    for frame in reference_frames:
        logging.info(
            f"Frame {frame_idx} : Drawing estimation landmarks onto image {ref_video_name} ..."
        )
        estimation = run_inference_no_crop(module, input_size=256, image=frame)
        output_frames_reference.append(estimation)
        frame_idx += 1

    output_frames_user = []
    frame_idx = 0
    for frame in user_frames:
        logging.info(
            f"Frame {frame_idx} : Drawing estimation landmarks onto image {usr_video_name} ..."
        )
        estimation = run_inference_no_crop(module, input_size=256, image=frame)
        output_frames_user.append(estimation)
        frame_idx += 1

    comp_frames = []

    # 4. Write Output and Upload

    # video_processor.write_frames_to_file(result_file_name, comp_frames)
    video_processor.write_frames_to_file(ref_video_name, output_frames_reference)
    video_processor.write_frames_to_file(usr_video_name, output_frames_user)

    logging.info("Uploading to S3...")
    # s3_client.upload_video(result_file_name)
    ref_video_name = "estimation_" + ref_video_name
    usr_video_name = "estimation_" + usr_video_name
    s3_client.upload_video(ref_video_name)
    s3_client.upload_video(usr_video_name)

    # Delete file after upload to S3
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    else:
        logging.warning("The output file does not exist")

    if os.path.exists(ref_video_name):
        os.remove(ref_video_name)
    if os.path.exists(usr_video_name):
        os.remove(usr_video_name)

    else:
        logging.warning("The output file does not exist")

    return {
        "file_names": {
            "reference": ref_video_name,
            "user": usr_video_name,
            "comparison": usr_video_name,
        },
        "all_scores": score_list,
        "mean_score": final_score,
    }
