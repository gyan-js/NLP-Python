
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model


train_data = pd.read_csv("D:/Kunal Programming/PYTHON/NLP-Python/PRO-C117-Project-Solution/static/assets/dataset/updated_product_dataset.csv")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "Text"]
    training_sentences.append(sentence)

model = load_model("D:/Kunal Programming/PYTHON/NLP-Python/PRO-C117-Project-Solution/static/assets/model/Customer_Review_Text_Emotion2.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

emo_code_url = {
    "Neutral": [0, "./static/assets/emoticons/neutral.png"],
    "Positive": [1,"./static/assets/emoticons/positive.png" ],
    "Negative": [2, "./static/assets/emoticons/negative.png"]
    }

def predict(text):

    predicted_emotion=""
    predicted_emotion_img_url=""
    
    if  text!="":
        sentence = []
        sentence.append(text)

        sequences = tokenizer.texts_to_sequences(sentence)

        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=-1)        
            
        for key, value in emo_code_url.items():
            if value[0]==predicted_class_label:
                predicted_emotion_img_url=value[1]
                predicted_emotion=key
        return predicted_emotion, predicted_emotion_img_url

print(predict('Great phone, do buy it. It has long lasting battery and beautiful display'))


