# Kannada Handwriting Recognition and Translation System 🖊️📜🔊

This project is a full-stack deep learning application that recognizes handwritten Kannada characters/words from uploaded images, translates the recognized text into English, and provides voice output in **both Kannada and English**.

---

## 📌 Features

* ✅ Upload handwritten Kannada image
* ✅ Recognizes the text using a trained Deep Learning model (CNN)
* ✅ Translates Kannada text to English
* ✅ Voice output for both Kannada and English
* ✅ User-friendly web interface

---

## 💠 Tech Stack

### 🔹 Frontend:

* HTML
* CSS
* JavaScript

### 🔹 Backend:

* Python
* Flask (Web framework)
* Deep Learning (CNN Model)

### 🔹 Additional:

* Google Text-to-Speech (gTTS) for voice output
* Translation API (e.g., Googletrans)
* Dataset: [Kaggle Kannada Handwritten Dataset](https://www.kaggle.com/)

---

## 📂 Project Structure

```
kannada-handwriting-recognition/
│
├── static/                # CSS, JS, image files
├── templates/             # HTML templates
├── model/                 # Trained CNN model files
├── app.py                 # Main Flask application
├── utils.py               # Image preprocessing and prediction logic
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

---

## 🚀 How to Run the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/kannada-handwriting-recognition.git
   cd kannada-handwriting-recognition
   ```

2. **Create virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app:**

   ```bash
   python app.py
   ```

5. **Open in browser:**
   Go to `http://127.0.0.1:5000` and upload a Kannada handwritten image.

---

## 🧐 Model Details

* **Model Type**: Convolutional Neural Network (CNN)
* **Framework**: TensorFlow / Keras
* **Trained on**: Kannada handwritten characters dataset from Kaggle
* **Accuracy**: \~ (Add your model accuracy here if known)

---

## 🔊 Voice Output

* Used **gTTS** to convert recognized Kannada and translated English text into speech.
* Plays audio automatically after prediction.

---

## 🌐 Translation

* Integrated **Google Translate** API using `googletrans` to convert Kannada text to English.

---

## 📸 Sample

> Upload an image like this:

![sample](static/sample_image.jpg)

> Output:

* Kannada: "ನಮಸ್ಕಾರ"
* English: "Hello"
* Voice Output: Plays both Kannada and English speech

---

## ✅ Future Improvements

* Add support for multiple lines and sentence-level recognition
* Improve accuracy using larger datasets or transformers
* Host on a cloud platform like Heroku or AWS

---

## 👨‍💻 Author

**Your Name**
Final Year MTech Student
Email: [your.email@example.com](mailto:your.email@example.com)

---

## 📄 License

This project is for academic purposes.
