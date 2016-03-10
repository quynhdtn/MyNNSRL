import argparse
import pickle
from liir.nlp.classifiers.Problem import Problem
from liir.nlp.classifiers.CRFModel import CRFModel
from liir.nlp.nnsrl.ml.DataSet import DataSet
from liir.nlp.nnsrl.ml.Instance import Instance
from liir.nlp.nnsrl.representation.Text import Text

__author__ = 'quynhdo'
class ArgumentClassifier:

    def __init__(self, N_fea_file= None, V_fea_file = None, we_list_file = None, temp_folder= "temp"):
        self.problems = {}
        self.temp_folder = temp_folder

        if N_fea_file is not None:
            nProb = Problem()
            nProb.loadFeatureConfig(N_fea_file, we_list_file)

            self.problems["N"] = nProb

        if V_fea_file is not None:
            vProb = Problem()
            vProb.loadFeatureConfig(V_fea_file, we_list_file)

            self.problems["V"] = vProb


    def use_CRF(self):
        self.problems['N'].used_sequence = True

        self.problems['V'].used_sequence = True
        crf1 = CRFModel()
        crf2 = CRFModel()

        self.problems['N'].model = crf1
        self.problems['V'].model = crf2

    def train_sequence(self, txt):
        dsV = DataSet()
        dsN = DataSet()
        for sen in txt:
            for p in sen.getPredicates():
                if p.pos.startswith("V"):
                    l=[]
                    for arg in sen:
                        if arg in p.arguments.keys():
                            ins = Instance((arg,p))
                            l.append((ins, p.arguments[arg]))
                    dsV.addSeq(l)
                else:
                    l=[]
                    for arg in sen:
                        if arg in p.arguments.keys():
                            ins = Instance((arg,p))
                            l.append((ins, p.arguments[arg]))
                    dsN.addSeq(l)

        self.problems['N'].train(dsN, self.temp_folder + "/N.mdl" )
        self.problems['V'].train(dsV, self.temp_folder + "/V.mdl")

    def predict_sequence(self, txt, outputN, outputV):
        dsV = DataSet()
        dsN = DataSet()
        for sen in txt:
            for p in sen.getPredicates():
                if p.pos.startswith("V"):
                    l=[]
                    for arg in sen:
                        if arg in p.arguments.keys():
                            ins = Instance((arg,p), False)
                            l.append((ins, p.arguments[arg]))
                    dsV.addSeq(l)
                else:
                    l=[]
                    for arg in sen:
                        if arg in p.arguments.keys():
                            ins = Instance((arg,p), False)
                            l.append((ins, p.arguments[arg]))
                    dsN.addSeq(l)


        rsN = self.problems['N'].predict(dsN)
        rsV = self.problems['V'].predict(dsV)

        rsNasList = []
        for t in rsN:
            for tt in t:
                rsNasList.append(tt)

        rsVasList = []
        for t in rsV:
            for tt in t:
                rsVasList.append(tt)

        f =open (outputN, "w")
        for v in rsNasList:
            f.write(str(v))
            f.write("\n")
        f.close()

        f =open (outputV, "w")
        for v in rsVasList:
            f.write(str(v))
            f.write("\n")
        f.close()

        return rsNasList, rsVasList

    def save(self, path_to_file):
        pickle.dump( self, open( path_to_file, "wb" ) )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-train", "--train", action="store_true")
    parser.add_argument("-test", "--test", action="store_true")

    parser.add_argument("one", help="Feature file for N if TRAIN or Model file if TEST")
    parser.add_argument("two", help="Feature file for V if TRAIN or Test file if TEST")
    parser.add_argument("three", help="We config if TRAIN or Output for N if TEST")
    parser.add_argument("four", help="Train file if TRAIN or Output for V if TEST")
    parser.add_argument("five",nargs='?', help="temp folder", default="temp")
    parser.add_argument("six",nargs='?', help="Model save path if TRAIN", default="srl.mdl")

    args = parser.parse_args()

    if args.train:


        ac = ArgumentClassifier(args.one,
                                args.two,
                               args.three,
                                args.four)

        txt = Text()
        txt.readConll2009Sentences(args.five)
        ac.use_CRF()
        ac.train_sequence(txt)
        ac.save(args.six)
    if args.test:


        txt.readConll2009Sentences(args.two)

        ac.predict_sequence(txt, args.three, args.four)
