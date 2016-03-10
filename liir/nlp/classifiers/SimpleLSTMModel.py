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


class SimpleLSTMModel(Model):
    def __init__(self, input_dim,maxlen, lstm_size, output_dim):
        Model.__init__(self)
        self.classifier = None
        self.maxlen=maxlen
        model = Sequential()
        model.add(Masking())
     #   model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(LSTM(lstm_size, input_dim=input_dim, return_sequences=True))  # try using a GRU instead, for fun
        #model.add(Dense(lstm_size, output_dim))
        model.add(TimeDistributedDense(output_dim=output_dim, input_dim=lstm_size))
        model.add(Activation('softmax'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='mean_squared_error', optimizer='rmsprop')


        self.classifier=model

    def train(self, X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True):
        self.classifier.fit(X_train, y_train, nb_epoch=  nb_epoch, batch_size=batch_size,  show_accuracy=show_accuracy)


    def evaluate(self, X_test, y_test, batch_size=16):
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)

        return self.model.evaluate(X_test, y_test, batch_size)

    def predict(self, X_test):
        return self.model.predict_classes(X_test)







