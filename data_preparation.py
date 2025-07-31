import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --- Config ---
image_dir = r"C:\Users\hp\Desktop\athees half\archive (4)"
csv_path = r"C:\Users\hp\Desktop\athees half\kan_labels.csv"

# --- Load and Clean Labels ---
label_df = pd.read_csv(csv_path).dropna()
label_df = label_df[label_df["Image_Path"].notnull()]

# --- Build Vocabulary ---
all_text = "".join(label_df["Label"].astype(str))
vocab = sorted(set(all_text))
char_to_num = {char: idx + 1 for idx, char in enumerate(vocab)}  # +1 for padding
num_to_char = {idx: char for char, idx in char_to_num.items()}

def text_to_sequence(text):
    return [char_to_num.get(ch, 0) for ch in text]

def preprocess_image(path):
    try:
        image = Image.open(path).convert('L')
        image = image.resize((128, 32))
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, axis=-1)
    except:
        return None

# --- Load Dataset ---
X, Y = [], []
for _, row in label_df.iterrows():
    img_path = os.path.join(image_dir, row["Image_Path"])
    label = str(row["Label"]).strip()
    img_array = preprocess_image(img_path)
    if img_array is not None and len(label) > 0:
        X.append(img_array)
        Y.append(text_to_sequence(label))

# --- Pad and Prepare ---
X = np.array(X)
Y = tf.keras.preprocessing.sequence.pad_sequences(Y, padding='post')
input_len = np.ones((len(X), 1)) * 32
label_len = np.array([[len(y)] for y in Y])

# --- Split ---
X_train, X_test, Y_train, Y_test, input_len_train, input_len_test, label_len_train, label_len_test = train_test_split(
    X, Y, input_len, label_len, test_size=0.1, random_state=42
)

# --- Save ---
np.savez("ocr_data.npz", 
         X_train=X_train, Y_train=Y_train,
         X_test=X_test, Y_test=Y_test,
         input_len_train=input_len_train, input_len_test=input_len_test,
         label_len_train=label_len_train, label_len_test=label_len_test)

import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump((char_to_num, num_to_char), f)

print("✅ Dataset and vocab prepared.")
