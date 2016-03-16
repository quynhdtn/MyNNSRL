import argparse
import pickle
from liir.nlp.classifiers.BidirectionalLSTMModel import BidirectionalLSTMModel
from liir.nlp.classifiers.Problem import Problem
from liir.nlp.classifiers.CRFModel import CRFModel
from liir.nlp.classifiers.S2SModel import S2SModel
from liir.nlp.nnsrl.ml.DataSet import DataSet
from liir.nlp.nnsrl.ml.Instance import Instance
from liir.nlp.nnsrl.representation.Text import Text
from liir.nlp.nnsrl.representation.Word import Predicate

__author__ = 'quynhdo'
class SRL:

    def __init__(self, fea_file = None, we_list_file = None, temp_folder= "temp"):
        self.prob=None
        self.temp_folder = temp_folder
        self.mdl_architecture = None
        self.label_map={}

        if fea_file is not None:
            nProb = Problem()
            nProb.loadFeatureConfig(fea_file, we_list_file)

            self.prob = nProb



    def train_lstm_test(self, txt, save_path="."):
        ds = DataSet()

        for sen in txt:
            for pred in sen.getPredicates():
                    l=[]
                    for w in sen:

                            ins = Instance((w,pred))
                            l.append((ins, pred.arguments[sen.index(w)]))
                    ds.addSeq(l)

        ml = ds.longestInstance()
        print("maxlen :")
        print(ml)
        X,Y = self.prob.getFeatureForTrainNumpy(ds)
        print ( "Finish processing data")
        self.label_map = ds.label_map
        #import pickle
        #with open(save_path+ "/model_fea.mdl", "wb") as f:
        #    pickle.dump(self,f)

        lstm = BidirectionalLSTMModel(input_dim=X[0].shape[1], maxlen=ml, lstm_size=128, output_dim=len(self.prob.label_map.keys()))
        self.mdl_architecture = (X[0].shape[1], ml, 128,len(self.prob.label_map.keys()) )
        import pickle
        with open(save_path+ "/model_fea.mdl", "wb") as f:
            pickle.dump(self,f)

        #self.prob.model = lstm
        lstm.train(X,Y, nb_epoch=1000)

        lstm.save_weights(save_path)




    def predict_lstm_test(self, txt):
        ds = DataSet()

        for sen in txt:
            for pred in sen.getPredicates():
                    l=[]
                    for w in sen:

                            ins = Instance((w,pred),used_for_training=False)

                            l.append((ins, pred.arguments[sen.index(w)]))
                    ds.addSeq(l)

        X,Y = self.prob.getFeatureForTestNumpy(ds)


        return self.prob.predict(X)


def loadSRL(mdl_path="."):
    file = open(mdl_path + "/"+"model_fea."
                               "mdl",'rb')
    srl = pickle.load(file)
    print(srl.mdl_architecture)
    lstm = BidirectionalLSTMModel(input_dim=int(srl.mdl_architecture[0]), maxlen=int(srl.mdl_architecture[1]),
                                  lstm_size=int(srl.mdl_architecture[2]), output_dim=int(srl.mdl_architecture[3]))


    lstm.classifier.load_weights(mdl_path + "/" + 'my_model_weights.h5')

    srl.prob.label_map=srl.label_map
    srl.prob.model=lstm
    return srl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()



    parser.add_argument("one", help="fea file")
    parser.add_argument("two", help="we file")
    parser.add_argument("three", help="temp folder")
    parser.add_argument("four", help="data file")
    parser.add_argument("five",nargs='?', help="temp folder", default="temp")
    parser.add_argument("six",nargs='?', help="Model save path if TRAIN", default="srl.mdl")

    args = parser.parse_args()
    #tg = SRL("/Users/quynhdo/Documents/WORKING/PhD/NewWorkspace/NNSRL/fea/fea2.config",
    #            "/Users/quynhdo/Documents/WORKING/PhD/NewWorkspace/NNSRL/fea/we.config", ".")

    tg = SRL(args.one,
               args.two, args.three)

    txt=Text()
    txt.readConll2005Sentences(args.four)
    #txt.readConll2005Sentences("/Users/quynhdo/Documents/WORKING/Data/conll2005/train-set")

   #txt.readConll2005Sentences("/Users/quynhdo/Documents/WORKING/Data/conll2005/conll05st-release/test.brown/test.brown.txt")

    tg.train_lstm_test(txt[0:10])

   #txt1=Text()
   #txt1.readConll2005Sentences("/Users/quynhdo/Documents/WORKING/Data/conll2005/conll05st-release/test.brown/test.brown.txt")
   #print (tg.predict_lstm_test(txt1))

   #txt2=Text()
   #txt2.readConll2005Sentences("/Users/quynhdo/Documents/WORKING/Data/conll2005/conll05st-release/test.wsj/test.wsj")
   #print (tg.predict_lstm_test(txt2))

   #srl = loadSRL("/Users/quynhdo/Documents/WORKING/PhD/NewWorkspace/NNSRL/liir/nlp/nnsrl/conll2005")
   #srl.predict_lstm_test(txt1[0:10])