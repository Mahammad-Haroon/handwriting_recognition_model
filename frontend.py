import streamlit as st
from PIL import Image
import numpy as np

# --- Modern CSS for a professional look ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif !important;
    background: #181c20 !important;
}
.stApp {
    background: linear-gradient(120deg, #232526 0%, #414345 100%) !important;
}
.main-card {
    background: #232b2b;
    border-radius: 18px;
    padding: 2.5em 2em 2em 2em;
    margin: 2em auto;
    max-width: 420px;
    box-shadow: 0 8px 32px rgba(56,142,60,0.18);
}
.stTitle, .stHeader, .stSubheader, .stCaption, .stMarkdown, .stText, .stDataFrame, .stTable {
    color: #fff !important;
}
.stFileUploader, .stFileUploader * {
    color: black !important;
    font-weight: bold !important;
    text-shadow: none !important;
}
.stFileUploader [data-testid="stFileDropzone"] {
    background: #020202 !important;
    border: 2px dashed #388e3c !important;
    border-radius: 12px !important;
    min-height: 120px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.stFileUploader [data-testid="stFileDropzone"] span,
.stFileUploader [data-testid="stFileDropzone"] div,
.stFileUploader [data-testid="stFileDropzone"] p,
.stFileUploader [data-testid="stFileDropzone"] label {
    color: #000000 !important;
    font-weight: bold !important;
}
.stFileUploader [data-testid="stFileUploaderUploadedFileDetails"],
.stFileUploader [data-testid="stFileUploaderUploadedFileDetails"] * {
    color: #000000 !important;
    font-weight: bold !important;
}
.stButton>button {
    background: linear-gradient(90deg, #388e3c 0%, #43a047 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    box-shadow: 0 2px 8px rgba(56,142,60,0.15);
    transition: background 0.2s;
    font-size: 1.1em;
    padding: 0.6em 1.5em;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #43a047 0%, #388e3c 100%) !important;
}
.result-card {
    background: #222c2a;
    border-radius: 12px;
    padding: 1.5em 2em;
    margin: 1.5em 0;
    box-shadow: 0 4px 24px rgba(56,142,60,0.12);
    color: #fff;
    font-size: 1.2em;
    font-weight: 500;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- Main UI ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.title("Kannada Handwriting OCR 🖋️")
st.write("Upload a handwritten Kannada word image to recognize the text.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    with st.spinner('Analyzing image...'):
        # --- Replace this with your model prediction logic ---
        import time
        time.sleep(2)
        kannada_word = "ಪಠ್ಯ"
        english_word = "Text"
        st.markdown(
            f'<div class="result-card">✅ <b>Kannada:</b> {kannada_word}<br><b>English:</b> {english_word}</div>',
            unsafe_allow_html=True
        )
        # --- Add voice output buttons if you want here ---

st.markdown('</div>', unsafe_allow_html=True)
