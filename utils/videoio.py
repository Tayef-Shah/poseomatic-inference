import cv2
import logging
import numpy as np
from moviepy.editor import ImageSequenceClip


class VideoIO:
    def __init__(self) -> None:
        pass

    def split_into_frames(self, video_url) -> np.ndarray:
        cap = cv2.VideoCapture(video_url)
        logging.info("Downloaded and captured S3 video")
        frames = []
        success, frame_image = cap.read()
        count = 0
        while success:
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            frames.append(frame_image)
            success, frame_image = cap.read()
            count += 1

        video = np.stack(frames, axis=0)
        cap.release()
        logging.info(f"Total frames: {count} --- Video shape: {video.shape}")

        return video

    def write_frames_to_file(self, file_name, frames) -> None:
        logging.info("Writing video file...")
        file_key = "estimation_" + file_name
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(file_key, audio=False)
