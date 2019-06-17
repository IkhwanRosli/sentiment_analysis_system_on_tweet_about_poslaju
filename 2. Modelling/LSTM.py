import tensorflow as tf
import numpy as np
import pandas as pd
import json
from keras.preprocessing.text import one_hot
import random
from keras.models import model_from_yaml
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

def embedded(data, v_size):
    from keras.preprocessing.text import one_hot
    dataList = data['text'].tolist()
    vocab_int = {}
    encoded =[]
    vocab_size = v_size
    for item in dataList:
        temp1 = (one_hot(item,vocab_size))
        temp2 = item.split()
        for i in range(len(temp2)):
            if temp2[i] in vocab_int:
                continue
            else:
                vocab_int[temp2[i]] = temp1[i]
        encoded.append(temp1)
    return encoded, vocab_int

#Modified the one-hot process in order to make sure each word has it own specific id.
def embedded2(data, v_size,vocab):
    dataList = data['text'].tolist()
    vocab_int = vocab
    encoded =[]
    vocab_size = v_size
    for item in dataList:
        notDone = True
        temp1 = []
        temp2 = []
        temp1 = (one_hot(item,vocab_size))
        temp2 = item.split()
        for i in range(len(temp2)):
            if temp2[i] in vocab_int.keys():
                pass
            else:
                while notDone:
                    if temp1[i] in vocab_int.values():
                        temp1[i] = random.randrange(1, vocab_size)
                    else:
                        notDone = False
            vocab_int[temp2[i]] = temp1[i]
        encoded.append(temp1)
    return encoded, vocab_int

test = pd.read_csv('data.csv',sep=';')
label = np.array(test['label'])

vocab_size = 200
review_int, vocab_int = embedded(test,vocab_size)

with open('vocab.json','w') as fp:
    json.dump(vocab_int , fp)

seq = 50
max_words = seq
features = np.zeros((len(review_int), seq), dtype=int)
for i,row in enumerate(review_int):
    features[i,-len(row):] = np.array(row)[:seq]

split_frac = 0.9
split_index = int(split_frac * len(features))
x_train, x_test = features[:split_index],features[split_index:]
y_train, y_test = label[:split_index],label[split_index:]

embedding_size=512
model = Sequential()
model.add(Embedding(vocab_size, embedding_size,input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics =['accuracy'])

batch = 32
epoch = 50

x_valid, y_valid = x_train[:batch], y_train[:batch]
x_train2,y_train2 = x_train[batch:], y_train[batch:]

model.fit(x_train2,y_train2, validation_data = (x_valid, y_valid), batch_size = batch, epochs = epoch)

scores = model.evaluate(x_test,y_test, verbose = 0 )
print("The Accuracy of the model is {}".format(scores[1]))

