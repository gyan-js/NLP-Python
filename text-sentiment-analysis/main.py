import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_raw_data = pd.read_excel(
    'D:/Kunal Programming/PYTHON/NLP-Python/text-sentiment-analysis/text-emotion-training-dataset.xlsx')
train_data = pd.DataFrame(train_raw_data["Text_Emotion"].str.split(
    ";", 1).to_list(), columns=["Text", "Emotion"])
train_data["Emotion"].unique()


training_sentences = []
training_labels = []

for i in range(10):
    sentence = train_data.loc[i, "Text"]
    training_sentences.append(sentence)

    labels = train_data.loc[i, "Emotion"]
    encoded_label = label_encoder.fit_transform([labels])[0]
    training_labels.append(encoded_label)


vocab_size = 10000
embedding_dim = 16
training_size = 20000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
training_sequence = tokenizer.texts_to_sequences(training_sentences)


padding_type = 'post'
max_length = 100
trunc_type = 'post'

train_padded = pad_sequences(training_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(train_padded[0:3])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=3),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=3),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(6, activation="softmax")
])



model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#num_epochs = 10
model.fit(train_padded, np.array(training_labels), epochs=10, verbose=2)

loss, acc = model.evaluate(train_padded, np.array(training_labels), verbose=2)

print("ACCURACY", acc * 100, '%  ')
print("LOSS", loss * 100 , '%')
