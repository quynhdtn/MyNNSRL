import pycrfsuite
from liir.nlp.nnsrl.features.FeatureGenerator import FeatureGenerator
from liir.nlp.nnsrl.ml.DataSet import DataSet
from liir.nlp.nnsrl.ml.Instance import Instance
from liir.nlp.nnsrl.representation.Text import Text

__author__ = 'quynhdo'

if __name__ == "__main__":
    fg = FeatureGenerator("/Users/quynhdo/Documents/WORKING/PhD/NewWorkspace/NNSRL/fea/fea.config","/Users/quynhdo/Documents/WORKING/PhD/NewWorkspace/NNSRL/fea/we.config")
    txt = Text()
    txt.readConll2009Sentences("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-evaluation-ood.txt")
    ds = DataSet()
    for sen in txt:
        for p in sen.getPredicates():

            if p.pos.startswith("V"):

                for arg in sen:
                    if arg in p.arguments.keys():
                        ins = Instance((arg,p))
                        ds.add(ins, p.arguments[arg])

    for f in fg:
        ds.extractFeature(f)

    idx = 0
    X=[]
    Y=[]
    for sen in txt:
        for p in sen.getPredicates():

            if p.pos.startswith("V"):
                sq_dt=[]
                sq_lbl=[]
                for arg in sen:
                    if arg in p.arguments.keys():
                        sq_dt.append(ds[idx][0].getIndices2(fg))
                        sq_lbl.append(ds[idx][1])
                        idx+=1


                iq= pycrfsuite.ItemSequence(sq_dt)

                X.append(iq)
                Y.append(sq_lbl)
    print("start training...")
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X, Y):
            trainer.append(xseq, yseq)
            trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
            })
    trainer.train("testV.mdl")


    tagger= pycrfsuite.Tagger()
    tagger.open("testV.mdl")
    Ypredict=[]
    for xseq in X:
                tagger.set(xseq)
                yseq= tagger.tag()
                for i in range(len(yseq)):
                    print(tagger.marginal(yseq[i],i))
                Ypredict.append(yseq)
    print(Ypredict)
    print (Y)
    print(len(Ypredict))
    print(len(Y))
    from sklearn.metrics import f1_score

    YasList=[]
    for YY in Y:
        for yl in YY:
            YasList.append(yl)

    YPasList=[]
    for YY in Ypredict:
        for yl in YY:
            YPasList.append(yl)





    print (f1_score( YasList,YPasList))
