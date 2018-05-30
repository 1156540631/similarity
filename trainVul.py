from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking, Bidirectional, Activation
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
#from keras.utils import to_categorical
from gensim.models import word2vec
#from sklearn.model_selection import train_test_split

import numpy as np
import os
import random
num_lstm = 64
num_dense = 64
rate_drop_lstm = 0.1
rate_drop_dense = 0.1
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 50
split_number = 16000
data = []
label = []
path = "newslices/"
for root, _, files in os.walk(path):
    for name in files:
        f = open(os.path.join(root, name), encoding="utf-8")
        tmp = f.read()
        data.append(tmp)
        label.append(int(name[-5]))
print("start tokenizer")

sequences = []
word_index = {}
dict_index = 1


for i in range(len(data)):
    word_start_index = 0
    for j in range(len(data[i])):
        if ((data[i][j] == ' ' or data[i][j] == '\n' or data[i][j] == '\t') and (j != word_start_index)):
            temp_word = data[i][word_start_index:j]
            word_start_index = j+1
            if (temp_word not in word_index):
                word_index[temp_word] = dict_index
                dict_index += 1
        elif ((data[i][j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (data[i][j] in '0123456789')or data[i][j] == '_'):
            word_start_index = word_start_index
        elif (j == len(data[i])-1 and ((data[i][j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (data[i][j] in '0123456789') or data[i][j] == '_')):
            temp_word = data[i][word_start_index:j+1]
            word_start_index = j+1
            if (temp_word not in word_index):
                word_index[temp_word] = dict_index
                dict_index += 1
        elif (data[i][j] in '!"#$%&()*+,-./:;<=>?@[\]^`{|}~'):
            if j>word_start_index:
                temp_word = data[i][word_start_index:j]
                if (temp_word not in word_index):
                    word_index[temp_word] = dict_index
                    dict_index += 1
            temp_word = data[i][j]
            word_start_index = j+1
            if (temp_word not in word_index):
                word_index[temp_word] = dict_index
                dict_index += 1
        else:
            word_start_index = j+1
#sequences
for i in range(len(data)):
    word_start_index = 0
    one_sequence = []
    for j in range(len(data[i])):
        if ((data[i][j] == ' ' or data[i][j] == '\n' or data[i][j] == '\t') and (j != word_start_index)):
            temp_word = data[i][word_start_index:j]
            word_start_index = j+1
            if (temp_word in word_index):
                one_sequence.append(word_index[temp_word])
        elif ((data[i][j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (data[i][j] in '0123456789')or data[i][j] == '_'):
            word_start_index = word_start_index
        elif (j == len(data[i])-1 and ((data[i][j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (data[i][j] in '0123456789') or data[i][j] == '_')):
            temp_word = data[i][word_start_index:j+1]
            word_start_index = j+1
            if (temp_word in word_index):
                one_sequence.append(word_index[temp_word])
        elif (data[i][j] in '!"#$%&()*+,-./:;<=>?@[\]^`{|}~'):
            if j>word_start_index:
                temp_word = data[i][word_start_index:j]
                if (temp_word in word_index):
                    one_sequence.append(word_index[temp_word])
            temp_word = data[i][j]
            word_start_index = j+1
            if (temp_word in word_index):
                one_sequence.append(word_index[temp_word])
        else:
            word_start_index = j+1
    sequences.append(one_sequence)

nb_words = len(word_index)+1

print('word_index is:',word_index)
print(sequences[0])
print('Found %s unique tokens.' % len(word_index))



print("start padding")
pad_sequence = []
pad_label = []
for i in range(len(sequences)):
    if len(sequences[i]) > MAX_SEQUENCE_LENGTH:
        continue
    w2v = []
    for j in range(len(sequences[i])):
        w2v.append(sequences[i][j])
    while len(w2v) < MAX_SEQUENCE_LENGTH:
        w2v.append(0)
    pad_sequence.append(w2v)
    pad_label.append(label[i])
print("padding OK")

print("length of pad_sequence:",len(pad_sequence))
print("length of pad label:",len(pad_label))
for i in range(len(pad_sequence)):
    for j in range(len(pad_sequence[i])):
        pad_sequence[i][j] = str(pad_sequence[i][j])
print("start word2vec")
model = word2vec.Word2Vec(pad_sequence, min_count=1, size=EMBEDDING_DIM)
model.save("word.model")

model = word2vec.Word2Vec.load("word.model")
print("creating enbedding index")
embeddings_index = {}
word_vectors = model.wv
for word, vocab_obj in model.wv.vocab.items():
    if int(vocab_obj.index) < nb_words:
        embeddings_index[word] = word_vectors[word]
print("creating embedding matrix")
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("example of sequence after padding",pad_sequence[0])

pos = []
neg = []
for i in range(len(pad_sequence)):
    if pad_label[i] == 0:
        neg.append(pad_sequence[i])
    else:
        pos.append(pad_sequence[i])
print("positive:",len(pos),"negative:",len(neg))
p_sample = random.sample(pos, 10000)
n_sample = random.sample(neg, 10000)
pad_sequence = []
pad_label = [1]*10000 + [0]*10000
pad_sequence = p_sample+n_sample
print("length of pad_label:",len(pad_label))
print("length of p_sample",len(p_sample))
print("length of n_sample",len(n_sample))



seed = 0
np.random.seed(seed)
np.random.shuffle(pad_sequence)
np.random.seed(seed)
np.random.shuffle(pad_label)

x_train = pad_sequence[:split_number]
y_train = pad_label[:split_number]
x_test = pad_sequence[split_number:]
y_test = pad_label[split_number:]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


def get_model():
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Masking(mask_value=0, input_shape=(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)))
    model.add(Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



model = get_model()
model.fit(x_train, y_train,
          batch_size=32,
          epochs=20,
          validation_data=(x_test, y_test))
model.save('single_lstm_model.h5')
