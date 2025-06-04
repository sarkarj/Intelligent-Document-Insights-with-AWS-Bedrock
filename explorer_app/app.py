# explorer_app/app.py

import streamlit as st
import boto3
import os
import tempfile
import tarfile
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("EMBEDDING_BUCKET")
REGION = os.getenv("AWS_REGION")
s3 = boto3.client("s3", region_name=REGION)

st.set_page_config(page_title="Vector Store Explorer", page_icon="📦")

# Refresh button in top-left with title
col1, col2 = st.columns([1, 10])
with col1:
    if st.button("🔄", help="Refresh Vector Store", key="refresh_button", use_container_width=True):
        st.session_state["refresh_vectors"] = True
with col2:
    st.markdown("### 📦 Vector Store Explorer")

if st.session_state.get("refresh_vectors", False):
    st.cache_data.clear()
    st.session_state["refresh_vectors"] = False
    st.markdown("<div style='color: green; font-size: 0.8rem;'>✅ Vector Store list refreshed!</div>", unsafe_allow_html=True)

@st.cache_data
def list_vector_archives(bucket):
    response = s3.list_objects_v2(Bucket=bucket)
    return [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".tar.gz")]

vector_files = list_vector_archives(BUCKET)
selected_file = st.selectbox("Select a vector store archive", vector_files)

if selected_file and st.button("Load and Inspect"):
    with st.spinner("Processing..."):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "vector_store.tar.gz")
                s3.download_file(BUCKET, selected_file, local_path)

                with tarfile.open(local_path, "r:gz") as tar:
                    st.info("Contents of archive:")
                    st.write(tar.getnames())
                    tar.extractall(path=tmpdir)

                index_dir = os.path.join(tmpdir, "faiss_index")
                if not os.path.exists(index_dir):
                    st.error("faiss_index folder not found in the archive.")
                else:
                    embeddings = FakeEmbeddings(size=1536)  # ✅ Adjust to match your original embedding dimension
                    vectordb = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)

                    st.success("Vector store loaded successfully!")
                    if hasattr(vectordb, "docstore") and hasattr(vectordb.docstore, "_dict"):
                        docs = list(vectordb.docstore._dict.values())
                        st.write(f"📌 Total chunks: {len(docs)}")
                        st.write("🔍 Sample chunk text:", docs[0].page_content[:500])
                    else:
                        st.warning("No document metadata found.")

                    if hasattr(vectordb, "index"):
                        st.write("📐 Embedding dimensionality:", vectordb.index.d)
                        st.write("🔢 Number of vectors:", vectordb.index.ntotal)

        except Exception as e:
            st.error(f"Error: {e}")