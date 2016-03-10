import argparse
import pickle
from keras.preprocessing import sequence
from sklearn import preprocessing
from liir.nlp.classifiers.CRFModel import CRFModel
from liir.nlp.classifiers.Problem import Problem
from liir.nlp.classifiers.SimpleLSTMModel import SimpleLSTMModel
from liir.nlp.nnsrl.ml.DataSet import DataSet
from liir.nlp.nnsrl.ml.Instance import Instance
from liir.nlp.nnsrl.representation.Text import Text
from liir.nlp.nnsrl.representation.Word import Word
from liir.nlp.nntagger.Seq2SeqTaggerModel import Seq2SeqTaggerModel
from liir.nlp.nntagger.io.TreeBankReader import readTreeBank, readBrown


from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout, Masking
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
import numpy as np
import numpy as np
__author__ = 'quynhdo'
class Tagger:

    def __init__(self, fea_file= None, we_list_file = None, temp_folder= "temp"):
        self.problems = {}
        self.temp_folder = temp_folder

        self.prob = Problem()
        self.prob.loadFeatureConfig(fea_file, we_list_file)




    def use_CRF(self):
        self.prob.used_sequence = True

        crf = CRFModel()

        self.prob.model = crf



    def use_S2S(self):
        self.prob.used_sequence = True

        s2s = Seq2SeqTaggerModel()

        self.prob.model = s2s

    def train_lstm(self, txt):
            sw = []
            ds = DataSet()
            for sen in txt:
                if len(sen) < 100:
                    ww = [1 for i in range(0,len(sen))]
                    for i in range(len(sen), 100):
                        ww.append(0)
                    #    w = Word()
                    #    w.form = "___NONE___"
                    #    w.word="___NONE___"
                    #    w.pos = "___NONE___"
                    #    w.sentence = sen
                    #    sen.append(w)
                    sw.append(ww)
                else:
                    ww = [1 for i in range(0,100)]
                    sw.append(ww)

            for sen in txt:
                        for w in sen:

                                ins = Instance((w))
                                ds.add(ins, w.pos)


            X,Y = self.prob.getFeatureForTrain(ds)
            print(X.shape)
            lb = preprocessing.LabelBinarizer()
            lb.fit(Y)
            Ygold=Y
            Y=lb.transform(Y)

            print(Y)
            X_train=[]

            Y_train=[]
            i=0
            for sen in txt:
                    l=[]
                    ll=[]
                    for w in sen:
                        l.append(X[i,:].tolist())

                        ll.append(Y[i])
                        i+=1

                    Y_train.append(ll)
                    X_train.append(l)

            X_train=np.array(X_train)
            Y_train=np.array(Y_train)
            print (X_train.shape)
        #    X_train.reshape(s[0],s[2])

            X_train = self.pad_sequences(X_train, maxlen=100)
            Y_train = sequence.pad_sequences(Y_train, maxlen=100, padding='post')


            X_train =  X_train.reshape(X_train.shape[0],X_train.shape[1], X_train.shape[3])
          #  Y_train =  Y_train.reshape(Y_train.shape[0],Y_train.shape[1], Y_train.shape[3])


            print (Y_train)
            print("X: ")

          #  lstm = SimpleLSTMModel(input_dim=X_train.shape[2],maxlen=100, lstm_size=128, output_dim=len(list(np.unique(Y))))
          #  lstm.train(X_train,Y_train)
            model = Sequential()
            model.add(Masking(input_shape=(100,X_train.shape[2]) ))

            #model.add(Embedding(nb_word, 128))
            model.add(LSTM(500, input_dim=X_train.shape[2],   return_sequences=True))
            model.add(TimeDistributedDense( input_dim=500, output_dim=28))
            model.add(Activation('softmax'))

            rms = RMSprop()
            model.compile(loss='categorical_crossentropy', optimizer=rms,sample_weight_mode="temporal")

            model.fit(X_train, Y_train, batch_size=10, nb_epoch=200, show_accuracy=True, sample_weight=np.asarray(sw))

            res = model.predict_classes(X_train)

            y_pred=[]
            for i in range(len(txt)):
                for v in res[i,0:len(txt[i])].tolist():
                    y_pred.append(v)

            y_gold=[]
            for yy in Y_train.tolist():
                for y in yy:
                    if 1 in y:
                        y_gold.append(y.index(1))

            from  sklearn.metrics import accuracy_score
            print (y_pred)
            print(y_gold)
            print (accuracy_score(y_gold,y_pred))




    def train_sequence(self, txt, use_lstm=False):
        ds = DataSet()

        for sen in txt:
                    l=[]
                    for w in sen:

                            ins = Instance(w)
                            l.append((ins, w.pos))
                    ds.addSeq(l)


        self.prob.train(ds, self.temp_folder + "/tag.mdl" )


    def predict_sequence(self, txt, output):
        ds = DataSet()

        for sen in txt:
            l=[]
            for w in sen:

                    ins = Instance(w)
                    l.append((ins, w.pos))
            ds.addSeq(l)

        rs = self.prob.predict(ds)

        rsasList = []
        for t in rs:
            for tt in t:
                rsasList.append(tt)


        f =open (output, "w")
        for v in rsasList:
            f.write(str(v))
            f.write("\n")
        f.close()


        from sklearn.metrics import f1_score

        YasList=[]
        for ins in ds:
            for inss in ins:
                YasList.append(inss[1])


        print (f1_score(rsasList, YasList))
        return rsasList

    def save(self, path_to_file):
        pickle.dump( self, open( path_to_file, "wb" ) )


    def pad_sequences(self, arr, maxlen):
        newarr=[]
        for sen in arr.tolist():
            s = sen
            if len(sen) < maxlen:
                for i in range(len(sen), maxlen):

                    s.append(np.zeros((1,len(s[0][0]))))

            newarr.append(s)
        return np.asarray(newarr)



if __name__=="__main__":

    tg = Tagger("/Users/quynhdo/Documents/WORKING/PhD/NewWorkspace/NNSRL/fea/posfea2",
                "/Users/quynhdo/Documents/WORKING/PhD/NewWorkspace/NNSRL/fea/we.config", ".")
  #  txt = readTreeBank()[0:3000]

    txt=Text()
    txt.readConll2009SentencesPOS("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-evaluation-ood.txt")
    '''
    tg.use_CRF()
    tg.train_sequence(txt[0:10])
    tg.save("posConll2009SG300.mdl")


    tg = pickle.load( open("postag.mdl", "rb" ))
    txttest=Text()
    txttest.readConll2009SentencesPOS("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-evaluation.txt")

    Ypredict = tg.predict_sequence(txt[0:10],"out.txt")
    print(Ypredict)
    '''

    tg.train_lstm(txt[0:10])
