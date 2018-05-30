from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking, Bidirectional
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
#from keras.utils import to_categorical
from gensim.models import word2vec
#from sklearn.model_selection import train_test_split
import numpy as np
import os
import random

judge_score = 50
#num_test_data实际上是所用测试数据的一半
num_test_data = 100

model = load_model('bilstm_model.h5')
MAX_SEQUENCE_LENGTH = 300
data = []
label = []
test_data = []
test_label = []
path = "newslices/"
testpath = "testslices/"
for root, _, files in os.walk(path):
    for name in files:
        f = open(os.path.join(root, name), encoding="utf-8")
        tmp = f.read()
        data.append(tmp)
        label.append(int(name[-5]))
print("start tokenizer")
#tokenizer = Tokenizer(char_level=False,split=" ")
#tokenizer.fit_on_texts(data)
#sequences = tokenizer.texts_to_sequences(data)
#word_index = tokenizer.word_index
sequences = []
test_sequences = []
#dictionary
word_index = {}
dict_index = 1
#establish the dictionary
print("length of data",len(data))
#print(data[0][0:4])
#print(data[0],'\n',data[1])
#print(type(data[0][0:4]))

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

#print('word_index is:',word_index)
#print(sequences[0])
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
for i in range(len(pad_sequence)):
    for j in range(len(pad_sequence[i])):
        pad_sequence[i][j] = str(pad_sequence[i][j])


pos = []
neg = []
for i in range(len(pad_sequence)):
    if pad_label[i] == 0:
        neg.append(pad_sequence[i])
    else:
        pos.append(pad_sequence[i])

p_sample = random.sample(pos, num_test_data)
n_sample = random.sample(neg, num_test_data)
#从数据集中随机抽2000个
target_sequence = []
target_label = []
for i in range(num_test_data):
    target_sequence.append(p_sample[i])
    target_label.append(1)
for i in range(num_test_data):
    target_sequence.append(n_sample[i])
    target_label.append(0)

#process of test data
for root, _, files in os.walk(testpath):
    for name in files:
        f = open(os.path.join(root, name), encoding="utf-8")
        tmp = f.read()
        test_data.append(tmp)
        test_label.append(int(name[-5]))

for i in range(len(test_data)):
    word_start_index = 0
    one_sequence = []
    for j in range(len(test_data[i])):
        if ((test_data[i][j] == ' ' or test_data[i][j] == '\n' or test_data[i][j] == '\t') and (j != word_start_index)):
            temp_word = test_data[i][word_start_index:j]
            word_start_index = j+1
            if (temp_word in word_index):
                one_sequence.append(word_index[temp_word])
        elif ((test_data[i][j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (test_data[i][j] in '0123456789')or test_data[i][j] == '_'):
            word_start_index = word_start_index
        elif (j == len(test_data[i])-1 and ((test_data[i][j] in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') or (test_data[i][j] in '0123456789') or test_data[i][j] == '_')):
            temp_word = test_data[i][word_start_index:j+1]
            word_start_index = j+1
            if (temp_word in word_index):
                one_sequence.append(word_index[temp_word])
        elif (test_data[i][j] in '!"#$%&()*+,-./:;<=>?@[\]^`{|}~'):
            if j>word_start_index:
                temp_word = test_data[i][word_start_index:j]
                if (temp_word in word_index):
                    one_sequence.append(word_index[temp_word])
            temp_word = test_data[i][j]
            word_start_index = j+1
            if (temp_word in word_index):
                one_sequence.append(word_index[temp_word])
        else:
            word_start_index = j+1
    test_sequences.append(one_sequence)

print("start padding")
test_pad_sequence = []
test_pad_label = []
for i in range(len(test_sequences)):
    if len(test_sequences[i]) > MAX_SEQUENCE_LENGTH:
        continue
    w2v = []
    for j in range(len(test_sequences[i])):
        w2v.append(test_sequences[i][j])
    while len(w2v) < MAX_SEQUENCE_LENGTH:
        w2v.append(0)
    test_pad_sequence.append(w2v)
    test_pad_label.append(test_label[i])
print("padding OK")
for i in range(len(test_pad_sequence)):
    for j in range(len(test_pad_sequence[i])):
        test_pad_sequence[i][j] = str(test_pad_sequence[i][j])
print("number of test_pad_sequence",len(test_pad_sequence))

def vul_or_not(i):
    pairs_1 = []
    pairs_2 = []
    score = []
    for j in range(len(test_pad_sequence)):
        pairs_1.append(target_sequence[i])
        pairs_2.append(test_pad_sequence[j])
    pairs_1 = np.array(pairs_1)
    pairs_2 = np.array(pairs_2)
    score_list = model.predict([pairs_1,pairs_2], batch_size=1, verbose=0)
    score_list = score_list.reshape(-1)
    print(score_list)
#    print(type(score_list))
#    for i in range(len(score_list)):
#        if score_list[i]>0.5:
#            score_list[i] = 1
#        elif score_list[i]<=0.5:
#            score_list[i] = 0
#    print(score_list)
    score = sum(score_list)
    print("score is:",score)
    if score>judge_score:
        return 1
    else:
        return 0
#正样本为有漏洞：P 负样本为无漏洞：N
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(target_sequence)):
    result = vul_or_not(i)
    if (result == 1 and target_label[i] == 1): 
        TP += 1
        print(i," real:1,  predict:1, TP\n")
    elif (result == 1 and target_label[i] == 0):
        FP += 1
        print(i," real:0,  predict:1, FP\n")
    elif (result == 0 and target_label[i] == 0):
        TN += 1
        print(i," real:0,  predict:0, TN\n")
    elif (result == 0 and target_label[i] == 1):
        FN += 1
        print(i," real:1,  predict:0, FN\n")
#FPR为误报率 TPR为真正类率
accuracy = (TP+TN)/(float(TP+TN+FP+FN))
TPR = TP/(float(TP+FN))
FPR = FP/(float(FP+TN))

print(" TP=",TP," TN=",TN," FP=",FP," FN=",FN)
print("\n final TPR is :",TPR)
print("\n final FPR is :",FPR)
print("\n final test accuracy is :",accuracy)