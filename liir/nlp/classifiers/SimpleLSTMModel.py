from keras.preprocessing import sequence
from liir.nlp.classifiers.Model import Model
from keras.optimizers import adam

__author__ = 'quynhdo'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, TimeDistributedDense, Masking
from keras.layers import Embedding
from keras.layers import LSTM



import os
__author__ = 'quynhdo'

import numpy as np
class SimpleLSTMModel(Model):
    def __init__(self, input_dim,maxlen, lstm_size, output_dim):
        Model.__init__(self)
        self.classifier = None
        self.maxlen=maxlen
        model = Sequential()
        model.add(Masking(input_shape=(maxlen, input_dim) ))
     #   model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(LSTM(lstm_size, input_dim=input_dim, return_sequences=True))  # try using a GRU instead, for fun
        #model.add(Dense(lstm_size, output_dim))
        model.add(TimeDistributedDense(output_dim=output_dim, input_dim=lstm_size))
        model.add(Activation('softmax'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',sample_weight_mode="temporal")
        self.classifier=model


    def train(self, X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True):
        '''

        :param X_train: list of matrix
        :param y_train: list of 2D array
        :param nb_epoch:
        :param batch_size:
        :param show_accuracy:
        :return:
        '''
        sw = []
        for s in X_train:
            if (s.shape[0]< self.maxlen):
                ww = [1 for i in range(0,s.shape[0])]
                for i in range(s.shape[0], self.maxlen):
                        ww.append(0)
                sw.append(ww)
            else:
                ww = [1 for i in range(0,self.maxlen)]
                sw.append(ww)

        X =  self.pad_sequences(X_train, self.maxlen)
        print (X.shape)
        Y_train = sequence.pad_sequences(y_train, maxlen=self.maxlen, padding='post')
        Y_train=Y_train.reshape(Y_train.shape[0],Y_train.shape[1], Y_train.shape[3])
        print(Y_train.shape)

        self.classifier.fit(X, Y_train, nb_epoch=  nb_epoch, batch_size=batch_size,  show_accuracy=show_accuracy, sample_weight=np.asarray(sw))


    def evaluate(self, X_test, y_test, batch_size=16):
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)

        return self.classifier.evaluate(X_test, y_test, batch_size)

    def predict(self, X_test, classes = None):
        X =  self.pad_sequences(X_test, self.maxlen)
        rs = self.classifier.predict_classes(X)
        print (rs)
        if classes is not None:
            rss= []
            for i in range(len(X_test)):
                rsss=[]
                for j in range(X_test[i].shape[0]):
                    v = rs[i][j]

                    for k in classes.keys():
                            if classes[k] == v:
                                rsss.append(k)
                                break
                rss.append(rsss)
            rs = rss
        return rs







    def pad_sequences(self, arr, maxlen):
        newarr=[]
        for sen in arr:
            s = sen.tolist()
            if len(sen) < maxlen:
                for i in range(len(sen), maxlen):
                    s.append(np.zeros((len(s[0]))))

            newarr.append(s)
        return np.asarray(newarr)


