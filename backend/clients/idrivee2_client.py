"""
iDrive E2 Storage Client
S3-compatible cloud storage using boto3
"""

import boto3
from typing import Optional, BinaryIO
from botocore.exceptions import ClientError
from app.settings import settings
from app.logger import logger


class IDriveE2Client:
    """Client for iDrive E2 cloud storage operations"""

    def __init__(self):
        """Initialize iDrive E2 client with boto3"""
        self.endpoint_url = settings.IDRIVEE2_ENDPOINT_URL
        self.access_key = settings.IDRIVEE2_ACCESS_KEY_ID
        self.secret_key = settings.IDRIVEE2_SECRET_ACCESS_KEY
        self.bucket_name = settings.IDRIVEE2_BUCKET_NAME

        # Initialize S3 client
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1'  # iDrive E2 doesn't strictly require region
        )

        logger.info(f"✅ iDrive E2 client initialized for bucket: {self.bucket_name}")

    def upload_file(
        self,
        file_obj: BinaryIO,
        object_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to iDrive E2 storage

        Args:
            file_obj: File object to upload
            object_name: S3 object name (key) in the bucket
            content_type: MIME type of the file

        Returns:
            str: Public URL of the uploaded file

        Raises:
            Exception: If upload fails
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self.client.upload_fileobj(
                file_obj,
                self.bucket_name,
                object_name,
                ExtraArgs=extra_args
            )

            # Construct file URL
            file_url = f"{self.endpoint_url}/{self.bucket_name}/{object_name}"

            logger.info(f"✅ File uploaded successfully: {object_name}")
            return file_url

        except ClientError as e:
            logger.error(f"❌ Failed to upload file {object_name}: {str(e)}")
            raise Exception(f"Failed to upload file: {str(e)}")

    def download_file(self, object_name: str) -> bytes:
        """
        Download a file from iDrive E2 storage

        Args:
            object_name: S3 object name (key) in the bucket

        Returns:
            bytes: File content as bytes

        Raises:
            Exception: If download fails
        """
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_name
            )

            file_content = response['Body'].read()
            logger.info(f"✅ File downloaded successfully: {object_name}")
            return file_content

        except ClientError as e:
            logger.error(f"❌ Failed to download file {object_name}: {str(e)}")
            raise Exception(f"Failed to download file: {str(e)}")

    def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from iDrive E2 storage

        Args:
            object_name: S3 object name (key) in the bucket

        Returns:
            bool: True if deletion was successful

        Raises:
            Exception: If deletion fails
        """
        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_name
            )

            logger.info(f"✅ File deleted successfully: {object_name}")
            return True

        except ClientError as e:
            logger.error(f"❌ Failed to delete file {object_name}: {str(e)}")
            raise Exception(f"Failed to delete file: {str(e)}")

    def list_files(self, prefix: Optional[str] = None) -> list:
        """
        List files in the bucket

        Args:
            prefix: Optional prefix to filter objects

        Returns:
            list: List of object keys

        Raises:
            Exception: If listing fails
        """
        try:
            kwargs = {'Bucket': self.bucket_name}
            if prefix:
                kwargs['Prefix'] = prefix

            response = self.client.list_objects_v2(**kwargs)

            if 'Contents' not in response:
                return []

            files = [obj['Key'] for obj in response['Contents']]
            logger.info(f"✅ Listed {len(files)} files")
            return files

        except ClientError as e:
            logger.error(f"❌ Failed to list files: {str(e)}")
            raise Exception(f"Failed to list files: {str(e)}")

    def file_exists(self, object_name: str) -> bool:
        """
        Check if a file exists in the bucket

        Args:
            object_name: S3 object name (key) in the bucket

        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=object_name
            )
            return True
        except ClientError:
            return False

    def get_file_url(self, object_name: str) -> str:
        """
        Get the URL for a file in the bucket

        Args:
            object_name: S3 object name (key) in the bucket

        Returns:
            str: Public URL of the file
        """
        return f"{self.endpoint_url}/{self.bucket_name}/{object_name}"

    def generate_presigned_url(
        self,
        object_name: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate a presigned URL for temporary file access

        Args:
            object_name: S3 object name (key) in the bucket
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            str: Presigned URL

        Raises:
            Exception: If URL generation fails
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_name
                },
                ExpiresIn=expiration
            )

            logger.info(f"✅ Presigned URL generated for: {object_name}")
            return url

        except ClientError as e:
            logger.error(f"❌ Failed to generate presigned URL: {str(e)}")
            raise Exception(f"Failed to generate presigned URL: {str(e)}")


# Singleton instance
_idrivee2_client: Optional[IDriveE2Client] = None


def get_idrivee2_client() -> IDriveE2Client:
    """
    Get or create IDriveE2Client singleton instance

    Returns:
        IDriveE2Client: Singleton client instance
    """
    global _idrivee2_client
    if _idrivee2_client is None:
        _idrivee2_client = IDriveE2Client()
    return _idrivee2_client
