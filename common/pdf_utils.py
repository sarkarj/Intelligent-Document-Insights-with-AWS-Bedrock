#common/pdf_utils.py
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import io

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text.strip())
    return "\n\n".join(texts)

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)