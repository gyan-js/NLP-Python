import sys
from keras.preprocessing.text import Tokenizer

def tokenize_text(index):
    trainning_sentences = []
    vocab_size  = 10000
    embeding_dim = 16
    oov_token = "<OOV>"
    training_size = 20000
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(trainning_sentences)
    training_sequence = tokenizer.texts_to_sequences(trainning_sentences)
    print(training_sequence[index])

sys.modules[__name__] = tokenize_text
    

