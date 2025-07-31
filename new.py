import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, 
                                     Dropout, GRU, Bidirectional, BatchNormalization, Input, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Step 1: Data Loading and Preprocessing
def preprocess_image(image_path, target_size=(128, 32)):
    img = load_img(image_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize
    return img_array

# Adjust dataset paths and load data
root_path = r'C:\Users\hp\Desktop\athees half\archive (4)'
csv_path = os.path.join(root_path, 'kan_labels.csv')
data = pd.read_csv(csv_path)
data['Image_Path'] = data['Image_Path'].apply(lambda x: os.path.join(root_path, x))
data['Processed_Images'] = data['Image_Path'].apply(preprocess_image)

# Check if labels are being fetched
def check_labels(data):
    for i, row in data.iterrows():
        print(f"Label for image {row['Image_Path']}: {row['Label']}")
        if i >= 5:  # Check the first 5 labels
            break
check_labels(data)

# Encode labels
def encode_label(label, max_len, num_classes):
    encoded = np.zeros((max_len, num_classes))
    for i, char in enumerate(label[:max_len]):
        encoded[i, ord(char) % num_classes] = 1
    return encoded

max_label_len = 32  # Adjusted to match the model's output sequence length
num_classes = 70  # Example: Adjust as needed for Kannada characters
data['Encoded_Labels'] = data['Label'].apply(lambda x: encode_label(x, max_label_len, num_classes))

# Prepare train/test splits
X = np.stack(data['Processed_Images'].values)
y = np.stack(data['Encoded_Labels'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Definition
def build_model(input_shape, output_size):
    inputs = Input(shape=input_shape)

    # CNN for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # Reshape for RNN
    new_seq_len = input_shape[0] // 4  # Adjust for pooling layers
    num_features = (input_shape[1] // 4) * 64  # Features per timestep after CNN layers
    x = tf.keras.layers.Reshape((new_seq_len, num_features))(x)

    # RNN for sequence modeling
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)

    # Output layer
    outputs = TimeDistributed(Dense(output_size, activation='softmax'))(x)

    return Model(inputs, outputs)

input_shape = (128, 32, 1)  # Correct input shape for grayscale images
model = build_model(input_shape=input_shape, output_size=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Ensure labels match output sequence length
def adjust_labels_for_output(y, target_seq_len):
    adjusted_labels = np.zeros((y.shape[0], target_seq_len, y.shape[2]))
    for i in range(y.shape[0]):
        adjusted_labels[i, :min(y.shape[1], target_seq_len), :] = y[i, :target_seq_len, :]
    return adjusted_labels

y_train_adjusted = adjust_labels_for_output(y_train, 32)  # Match output sequence length
y_test_adjusted = adjust_labels_for_output(y_test, 32)

# Step 3: Model Training
batch_size = 32
epochs = 30
history = model.fit(
    x=X_train,
    y=y_train_adjusted,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test_adjusted),
    verbose=2
)  

# Step 4: Save the Model (Recommended Keras format)
model_save_path = "athees.keras"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")