from keras.optimizers import adam

__author__ = 'quynhdo'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaskedLayer
from keras.layers import Embedding
from keras.layers import LSTM



class LSTMSeqClassification:
    def __init__(self, input_dim,maxlen, lstm_size, output_dim):
        model = Sequential()
     #   model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(LSTM(lstm_size, dropout_W=0.5, dropout_U=0.1, input_dim=input_dim, input_length=maxlen))  # try using a GRU instead, for fun
        model.add(Dropout(0.5))
        model.add(Dense(output_dim))
        model.add(Activation('softmax'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam, class_mode="categorical")


        self.model=model

    def fit(self, X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True):
        self.model.fit(X_train, y_train, nb_epoch=  nb_epoch, batch_size=batch_size,  show_accuracy=show_accuracy)


    def evaluate(self, X_test, y_test, batch_size=16):
        return self.model.evaluate(X_test, y_test, batch_size)

    def predict(self, X_test):
        return self.model.predict_classes(X_test)


