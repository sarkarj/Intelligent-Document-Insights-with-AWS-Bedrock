# admin_app/app.py

import streamlit as st
import sys
import os

# Add the root directory to sys.path so we can import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embed_utils import process_and_upload_pdf

st.title("📄 Admin: Upload & Embed Documents")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and st.button("Process & Upload"):
    with st.spinner("Processing..."):
        file_name = uploaded_file.name
        success = process_and_upload_pdf(uploaded_file, file_name)
        if success:
            st.success(f"Uploaded and embedded: {file_name}")
        else:
            st.error("Failed to process the file.")