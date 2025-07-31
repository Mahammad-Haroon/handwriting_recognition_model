import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
import os
import base64
from googletrans import Translator

# Set page title
st.set_page_config(page_title="Kannada Handwritten Character Recognition", layout="centered")

# Set background image with dark overlay using custom CSS
def set_bg_hack(main_bg):
    main_bg_ext = "jpg"
    try:
        with open(main_bg, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        page_bg_img = f'''
        <style>
        .stApp {{
          background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                      url("data:image/{main_bg_ext};base64,{encoded_string}") !important;
          background-size: cover !important;
          background-position: center !important;
          color: white !important;
        }}
        .stMarkdown, .stTextInput, .stFileUploader, .stButton, .stTitle, .stSpinner, .stSuccess, .stError, .stImage, .stColumn, .stAlert, .stText, .stSubheader, .stCaption, .stHeader, .stDataFrame, .stTable, .stRadio, .stCheckbox, .stSelectbox, .stSlider, .stNumberInput, .stDateInput, .stTimeInput, .stColorPicker, .stForm, .stFormSubmitButton, .stExpander, .stMetric, .stJson, .stCode, .stException, .stHelp, .stTooltip, .stDownloadButton, .stProgress, .stDivider, .stTabs, .stTab, .stContainer, .stVerticalBlock, .stHorizontalBlock, .stSidebar, .stSidebarContent, .stSidebarHeader, .stSidebarFooter, .stSidebarTitle, .stSidebarSubheader, .stSidebarCaption, .stSidebarText, .stSidebarMarkdown, .stSidebarImage, .stSidebarButton, .stSidebarRadio, .stSidebarCheckbox, .stSidebarSelectbox, .stSidebarSlider, .stSidebarNumberInput, .stSidebarDateInput, .stSidebarTimeInput, .stSidebarColorPicker, .stSidebarForm, .stSidebarFormSubmitButton, .stSidebarExpander, .stSidebarMetric, .stSidebarJson, .stSidebarCode, .stSidebarException, .stSidebarHelp, .stSidebarTooltip, .stSidebarDownloadButton, .stSidebarProgress, .stSidebarDivider, .stSidebarTabs, .stSidebarTab, .stSidebarContainer, .stSidebarVerticalBlock, .stSidebarHorizontalBlock {{
          color: white !important;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning('background.jpg not found in the app directory. Please add it for the background image to appear.')

set_bg_hack('background.jpg')

# Load model
model = tf.keras.models.load_model('handwriting_recognition_model.h5')  # Change if model filename is different

# Load label mapping from class_labels.npy (matches model training)
class_labels = np.load('class_labels.npy', allow_pickle=True)
label_dict = dict(enumerate(class_labels))

# Translator instance
translator = Translator()

# Preprocessing function (updated for shape (128, 32, 1))
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((32, 128))  # PIL uses (width, height)
    image_array = np.array(image).astype('float32') / 255.0  # Normalize
    image_array = image_array.reshape(1, 128, 32, 1)  # Match model input shape
    return image_array

# Streamlit interface
st.title("Kannada Handwritten Character Recognition 🖋️")
st.write("Upload a handwritten Kannada character image to recognize its corresponding label.")

# Custom CSS to make all file uploader text black, but uploaded file name and size white
st.markdown('''
<style>
    /* Make all file uploader text black by default */
    .stFileUploader, .stFileUploader * {
        color: black !important;
        font-weight: bold !important;
        text-shadow: none !important;
    }
    .stFileUploader [data-testid="stFileDropzone"] {
        background: rgba(30,30,30,0.85) !important;
        border-radius: 8px !important;
    }
    /* Make uploaded file name and size white (most specific, placed last) */
    .stFileUploader [data-testid="stFileUploaderUploadedFileDetails"],
    .stFileUploader [data-testid="stFileUploaderUploadedFileDetails"] * {
        color: white !important;
    }
</style>
''', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    with st.spinner('Analyzing image...'):
        try:
            processed_image = preprocess_image(Image.open(uploaded_file))
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            predicted_label = label_dict.get(predicted_class, "Unknown")
            # Translate to English
            translation = translator.translate(str(predicted_label), src='kn', dest='en')
            translated_label = translation.text
            # Custom green background and white text for prediction result
            st.markdown(f'''
                <div style="background-color:#388e3c;padding:1em 1.5em;border-radius:8px;margin-bottom:1em;">
                    <span style="color:white;font-size:1.2em;font-weight:bold;">✅ Predicted Label: {predicted_label} (Kannada) → {translated_label} (English)</span>
                </div>
            ''', unsafe_allow_html=True)
            # Voice output (Kannada and English)
            tts_kn = gTTS(text=str(predicted_label), lang='kn')
            audio_file_kn = 'predicted_label_kn.mp3'
            tts_kn.save(audio_file_kn)
            tts_en = gTTS(text=str(translated_label), lang='en')
            audio_file_en = 'predicted_label_en.mp3'
            tts_en.save(audio_file_en)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Kannada Voice"):
                    audio_bytes_kn = open(audio_file_kn, 'rb').read()
                    st.audio(audio_bytes_kn, format='audio/mp3', start_time=0)
            with col2:
                if st.button("English Voice"):
                    audio_bytes_en = open(audio_file_en, 'rb').read()
                    st.audio(audio_bytes_en, format='audio/mp3', start_time=0)
            os.remove(audio_file_kn)
            os.remove(audio_file_en)
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
