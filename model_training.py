import numpy as np
import tensorflow as tf
import pickle

# Load Data
data = np.load("ocr_data.npz", allow_pickle=True)
X_train, Y_train = data['X_train'], data['Y_train']
input_len_train, label_len_train = data['input_len_train'], data['label_len_train']

with open("vocab.pkl", "rb") as f:
    char_to_num, num_to_char = pickle.load(f)

# Model
inputs = tf.keras.layers.Input(shape=(32, 128, 1), name='image_input')

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

new_shape = (8, 512)
x = tf.keras.layers.Reshape(target_shape=new_shape)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
x = tf.keras.layers.Dense(len(char_to_num)+1, activation='softmax')(x)

labels = tf.keras.layers.Input(name='label', shape=(None,), dtype='float32')
input_len = tf.keras.layers.Input(name='input_len', shape=(1,), dtype='int64')
label_len = tf.keras.layers.Input(name='label_len', shape=(1,), dtype='int64')

loss_out = tf.keras.layers.Lambda(lambda args: tf.keras.backend.ctc_batch_cost(*args))([labels, x, input_len, label_len])

model = tf.keras.models.Model(inputs=[inputs, labels, input_len, label_len], outputs=loss_out)
model.compile(optimizer='adam')

model.fit(
    x=[X_train, Y_train, input_len_train, label_len_train],
    y=np.zeros(len(X_train)),
    batch_size=64,
    epochs=20
)

model.save("kannada_ocr_model.h5")
print("✅ Model trained and saved.")
