from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
import numpy as np
maxlen = 100

batch_size = 1
nb_word = 4
nb_tag = 2

X_train = np.array( [[[1,2,3],[3,4,4]],[[1,3,3],[4,5,4]]])#two sequences, one is [1,2] and another is [1,3]
Y_train = [[[0,1],[1,0]],[[0,1],[1,0]]] #the output should be 3D and one-hot for softmax output with categorical_crossentropy
X_test =  np.array( [[[1,2,3],[3,4,4]],[[1,3,3],[4,5,4]]])
Y_test = [[[0,1],[1,0]],[[0,1],[1,0]]]
sw = [[1,1],[1,1]]
for i in range (0,98):
    sw[0].append(0)
    sw[1].append(0)

print (X_train.shape)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')

print (X_train)

Y_train = np.asarray(Y_train, dtype='int32')
print (Y_train)
Y_test = np.asarray(Y_test, dtype='int32')
print (Y_train.shape)
Y_train = sequence.pad_sequences(Y_train, maxlen=maxlen, padding='post')
Y_test = sequence.pad_sequences(Y_test, maxlen=maxlen, padding='post')
print (Y_train)
model = Sequential()
#model.add(Embedding(nb_word, 128))
model.add(LSTM(128, input_dim=3,  return_sequences=True))
model.add(TimeDistributedDense( input_dim=128, output_dim= nb_tag ))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms,sample_weight_mode="temporal")

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=100, show_accuracy=True,sample_weight=np.asarray(sw))
res = model.predict_classes(X_test)
print('res',res)
res=model.predict(X_test)
print('res',res)
