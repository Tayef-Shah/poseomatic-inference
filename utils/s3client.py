import logging
import boto3
from io import BytesIO
from PIL import Image


class S3Client:
    def __init__(self, region_name, bucket_name) -> None:
        self.region_name = region_name
        self.bucket = bucket_name
        self.s3 = boto3.client("s3", region_name=self.region_name)
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(logging.INFO)

    def load_image(self, file_key):
        response = self.s3.get_object(Bucket=self.bucket, Key=file_key)
        http_status = response["ResponseMetadata"]["HTTPStatusCode"]
        self.logger.info(f"S3 fetch response: {http_status}")
        image = Image.open(BytesIO(response["Body"].read()))
        return image

    def upload_image(self, image, image_path):
        file_path = "estimation_" + image_path

        with BytesIO() as output:
            image.save(output, format="JPEG")
            output.seek(0)
            # might want to catch response code
            self.s3.upload_fileobj(output, self.bucket, file_path)
