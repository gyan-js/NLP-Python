import pandas as pd
import numpy as np

import tensorflow
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
# train data
train_data = pd.read_csv("D:/Kunal Programming/PYTHON/NLP-Python/text-sentiment-analyzer/static/data_files/tweet_emotions.csv")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "content"]
    training_sentences.append(sentence)

#load model
model = load_model("D:/Kunal Programming/PYTHON/NLP-Python/text-sentiment-analyzer/static/model_files/Tweet_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

#assign emoticons for different emotions
emo_code_url = {
    "empty": [0, "./static/emoticons/Empty.png"],
    "sadness": [1,"./static/emoticons/Sadness.png" ],
    "enthusiasm": [2, "./static/emoticons/Enthusiasm.png"],
    "neutral": [3, "./static/emoticons/Neutral.png"],
    "worry": [4, "./static/emoticons/Worry.png"],
    "surprise": [5, "./static/emoticons/Surprise.png"],
    "love": [6, "./static/emoticons/Love.png"],
    "fun": [7, "./static/emoticons/fun.png"],
    "hate": [8, "./static/emoticons/hate.png"],
    "happiness": [9, "./static/emoticons/happiness.png"],
    "boredom": [10, "./static/emoticons/boredom.png"],
    "relief": [11, "./static/emoticons/relief.png"],
    "anger": [12, "./static/emoticons/anger.png"]
    
    }
# write the function to predict emotion

        
def predict(text):
    preicted_emotion = ""
    predicted_emotion_img_url = ""

    if text != "":
        sentence = []
        sentence.append(text)

        sequence = tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        testing_padded = np.array(padded)

        predicted_class_label = np.argmax(model.predict(testing_padded), axis=1)
        print(predicted_class_label)

        for key, value in emo_code_url.items():
            if value[0] == predicted_class_label:
                preicted_emotion = key
                predicted_emotion_img_url = value[1]
                
        return preicted_emotion, predicted_emotion_img_url