from keras.optimizers import SGD

__author__ = 'quynhdo'
# a MLP using Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import math

class MLP:
    def __init__(self, input_dim, hidden_layer_dims, activation_function="tanh"):

        if (len(hidden_layer_dims)==1):
            nHidden = int( math.sqrt(input_dim * (hidden_layer_dims[0]+2)) + 2 *math.sqrt(input_dim / (hidden_layer_dims[0]+2)) )
            hidden_layer_dims.insert(0, nHidden)
        model = Sequential()
        model.add(Dense(hidden_layer_dims[0], input_dim=input_dim, init='uniform')
                  )
        for i in range(1, len(hidden_layer_dims)):
            model.add(Activation(activation_function))
            model.add(Dropout(0.1))
            model.add(Dense(hidden_layer_dims[i], init='uniform'))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, class_mode="categorical")
        self.model = model

    def fit(self, X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True):
        self.model.fit(X_train, y_train, nb_epoch=  nb_epoch, batch_size=batch_size,  show_accuracy=show_accuracy)


    def evaluate(self, X_test, y_test, batch_size=16):
        return self.model.evaluate(X_test, y_test, batch_size)

    def predict(self, X_test):
        return self.model.predict_classes(X_test)


if __name__ == "__main__":

    from sklearn.datasets import load_iris
    import numpy
    iris = load_iris()
    X= iris.data
    Y = iris.target

    from sklearn import metrics
    import theano as th
    import theano.tensor as T
    from sklearn import preprocessing



    mp=MLP(input_dim =  len(X[0]),    hidden_layer_dims=[ len(set(Y))])
    from sklearn import cross_validation
    import scipy.sparse
    X_train, X_test, y_train, y_test = cross_validation.train_test_split( iris.data, Y, test_size=0.4, random_state=0)
   # X_train = scipy.sparse.csr_matrix( X_train)
#    X_test = scipy.sparse.csr_matrix(y_test)


    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)


    y_train= lb.transform(y_train)



    mp.fit(X_train, y_train, nb_epoch=3000, batch_size=30)


    y_pred = mp.predict(X_test)
  #  y_test = lb.transform(y_test)




    print (y_pred)

    print (metrics.f1_score(y_test,y_pred))


