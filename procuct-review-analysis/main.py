import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

dataframe = pd.read_excel(
    'D:/Kunal Programming/PYTHON/NLP-Python/procuct-review-analysis/updated_product_dataset.xlsx')



print(dataframe["Emotion"].unique())

encoded_emotions = {"Neutral": 0, "Positive": 1, "Negative": 2}
dataframe.replace(encoded_emotions, inplace = True)
print(dataframe.head(5))

training_sentences = []
training_labels = []
