# src/core/aws.py
"""AWS S3 helpers — used by the upload-to-S3 step in the ingestion entry points."""

import boto3


def get_s3_client(aws_access_key: str = None, aws_secret_key: str = None, region: str = None):
    if aws_access_key and aws_secret_key:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
        )
    return boto3.client('s3', region_name=region)


def upload_to_s3(local_file_path: str, bucket_name: str, s3_key: str, s3_client) -> bool:
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        return True
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        return False
