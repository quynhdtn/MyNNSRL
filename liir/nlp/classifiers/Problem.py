from liir.nlp.nnsrl.features.FeatureGenerator import FeatureGenerator
import pickle
import numpy as np
__author__ = 'quynhdo'

def LoadProblemFromFile(path_to_file):
    return pickle.load( open( path_to_file, "rb" ))

class Problem(object):
    def __init__(self, fg=None, used_sequence=False, model = None):
        self.fg = fg
        self.is_trained = False
        self.model = model
        self.labels = None
        self.used_sequence = used_sequence
        self.label_map={}

    def loadFeatureConfig (self, fea_config_file, we_config_file=None):
        self.fg = FeatureGenerator(fea_config_file, we_config_file)

    def train(self, ds, model_path=None):
        X,Y=ds.extractFeatureForMatrix(self.fg)
        print (X)
        print (Y)
        '''
        X,Y=None,None
        if self.used_sequence:
            X,Y = ds.asSequence(self.fg)

        else:
            X,Y = ds.asMatrix(self.fg)

            self.labels = list(np.unique(Y))
            Y = [str(self.labels.index(y)) for y in Y]


        self.model.train(X,Y, model_path)
        self.is_trained=True
        '''

    def getFeatureForTrain(self, ds, model_path=None):
        ds.extractFeatureForMatrix(self.fg)
        X,Y=None,None
        if self.used_sequence:
            X,Y = ds.asSequence(self.fg)

        else:
            X,Y = ds.asMatrix(self.fg)

            self.labels = list(np.unique(Y))
            Y = [str(self.labels.index(y)) for y in Y]
        return X,Y

    def getFeatureForTrainNumpy(self, ds):
        X,Y=ds.extractFeatureForMatrix(self.fg)

        self.label_map=ds.label_map

        return X,Y

    def getFeatureForTestNumpy(self, ds):
        ds.label_map = self.label_map

        X,Y=ds.extractFeatureForMatrix(self.fg, store_label_map=False)

        return X,Y


    def predictLstm(self, ds):
        ds.label_map = self.label_map
        X,Y= ds.extractFeatureForMatrix(self.fg)
        rs = self.model.predict(X, self.label_map)

        return rs


    def predict (self, ds):
        if not self.is_trained:
            return None
        ds.extractFeature(self.fg)
        X,Y=None,None
        if self.used_sequence:
            X,Y = ds.asSequence(self.fg)
        else:
            X,Y = ds.asMatrix(self.fg)

        return self.model.predict(X)

    def save(self, path_to_file):
        pickle.dump( self, open( path_to_file, "wb" ) )




