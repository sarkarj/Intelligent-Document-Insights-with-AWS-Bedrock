#common/s3_utils.py
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("EMBEDDING_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")

s3 = boto3.client("s3", region_name=AWS_REGION)

def upload_to_s3(file_path, key):
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET, key)

def list_tar_archives():
    """
    List all .tar.gz files from the specified S3 bucket.
    Returns a list of S3 keys for archives.
    """
    try:
        paginator = s3.get_paginator("list_objects_v2")
        archives = []

        for page in paginator.paginate(Bucket=S3_BUCKET):
            for obj in page.get("Contents", []):
                key = obj.get("Key", "")
                if key.endswith(".tar.gz"):
                    archives.append(key)

        return sorted(archives)

    except Exception as e:
        print(f"❌ Error listing archives from S3: {e}")
        return []