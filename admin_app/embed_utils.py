# admin_app/embed_utils.py
import sys
import os
import tempfile
import tarfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add root path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.pdf_utils import extract_text_from_pdf
from common.s3_utils import upload_to_s3

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
REGION = os.getenv("AWS_REGION")


def process_and_upload_pdf(file, file_name):
    try:
        # 1. Extract text
        text = extract_text_from_pdf(file)

        # 2. Split into semantically coherent chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)

        if not chunks:
            return False

        # 3. Wrap chunks in Document objects with metadata
        documents = [
            Document(page_content=chunk, metadata={"source": file_name, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]

        # 4. Generate embeddings and create FAISS index
        embeddings = BedrockEmbeddings(model_id=BEDROCK_MODEL_ID, region_name=REGION)
        vectordb = FAISS.from_documents(documents, embeddings)

        # 5. Save FAISS index to temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "faiss_index")
            vectordb.save_local(index_path)

            tar_path = os.path.join(tmpdir, f"{file_name}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(index_path, arcname="faiss_index")

            # 6. Upload to S3
            upload_to_s3(tar_path, f"{file_name}.tar.gz")

        return True

    except Exception:
        return False