import pycrfsuite
from liir.nlp.classifiers.Model import Model
import os
__author__ = 'quynhdo'


class CRFModel(Model):
    def __init__(self, c1=1.0, c2=1e-3, max_iterations=50,feature_possible_transitions=True):
        Model.__init__(self)
        self.classifier = None

        self.c1=c1
        self.c2=c2

        self.max_iterations= max_iterations
        self.feature_possible_transitions = feature_possible_transitions

    def train(self, X, Y, model_path=None):


        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params({
        'c1': self.c1,   # coefficient for L1 penalty
        'c2': self.c2,  # coefficient for L2 penalty
        'max_iterations':self.max_iterations,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': self.feature_possible_transitions
        })

        for xseq, yseq in zip(X, Y):
            trainer.append(xseq, yseq)
        if model_path is None:
            model_path = "crf.mdl"
        trainer.train(model_path)
        self.model_path = model_path



    def predict(self,XPred):
        if self.classifier is None:
            self.classifier = pycrfsuite.Tagger()
            self.classifier.open(self.model_path)

        Ypredict=[]
        for xseq in XPred:
                        self.classifier.set(xseq)
                        yseq= self.classifier.tag()
                      #  for i in range(len(yseq)):
                      #      print(tagger.marginal(yseq[i],i))
                        Ypredict.append(yseq)

        from collections import Counter
        info = self.classifier.info()

        def print_transitions(trans_features):
            for (label_from, label_to), weight in trans_features:
                print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

        print("Top likely transitions:")
        print_transitions(Counter(info.transitions).most_common(15))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(info.transitions).most_common()[-15:])
        return Ypredict
