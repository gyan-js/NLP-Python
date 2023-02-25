import pandas as pd
from keras.preprocessing.text  import Tokenizer

train_raw_data = pd.read_excel('text-emotion-training-dataset.xlsx')

train_data = pd.DataFrame(train_raw_data["Text_Emotion"].str.split(";", 1).to_list(), columns=["Text", "Emotion"])
train_data["Emotion"].unique()
'''
print(train_data.head(10))
print(train_data["Emotion"].unique())
'''

training_sentences = []
training_labels = []

for i in range(10):
    sentence = train_data.loc[i,"Text"]
    training_sentences.append(sentence)

    labels = train_data.loc[i, "Emotion"]
    training_labels.append(labels)
    print(sentence)
    print(labels)

vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
training_sequence = tokenizer.texts_to_sequences(training_sentences)

print(training_sequence[9])