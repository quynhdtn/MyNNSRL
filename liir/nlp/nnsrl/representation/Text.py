import re
from liir.nlp.nnsrl.representation.Sentence import SentenceConll2005, SentenceConll2009, SentenceConll2009POS

__author__ = 'quynhdo'

# this class is used to define  a text
class Text(list):
    def __init__(self):   # value should be  a list of Sentence
        list.__init__(self)

    def readConll2005Sentences(self, path):
        f = open(path, 'r')
        sens=[]
        words=[]

        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    sens.append(words)
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            sens.append(words)

        for sen in sens:
            conll2005sen = SentenceConll2005(sen)
            self.append(conll2005sen)

    def readConll2009Sentences(self, path):
        f = open(path, 'r')
        sens=[]
        words=[]

        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    sens.append(words)
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            sens.append(words)

        for sen in sens:
            conll2009sen = SentenceConll2009(sen)
            self.append(conll2009sen)

    def readConll2009SentencesPOS(self, path):
        f = open(path, 'r')
        sens=[]
        words=[]

        for l in f:
            match = re.match("\\s+", l)
            if match:
                if len(words) != 0:
                    sens.append(words)
                    words = []
            else:
                words.append(l.strip())
        if len(words) != 0:
            sens.append(words)

        for sen in sens:
            possen = SentenceConll2009POS(sen)
            self.append(possen)

    def getVob(self, type="all"):
        vob=set()
        if type == "all":
            for sen in self:
                for w in sen:
                    vob.add(w.word)

        if type == "pred":
            for sen in self:
                for p in sen.getPredicates():
                    vob.add(p.word)

        return vob






if __name__=="__main__":
    txt = Text()
    #txt.readConll2005Sentences("/Users/quynhdo/Documents/WORKING/Data/CoNLL2005/conll05st-release/test.brown/test.brown.txt")
    txt.readConll2009Sentences("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-train.txt")
    s1= txt.getVob()

    txt1 = Text()
    #txt.readConll2005Sentences("/Users/quynhdo/Documents/WORKING/Data/CoNLL2005/conll05st-release/test.brown/test.brown.txt")
    txt1.readConll2009Sentences("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-evaluation-ood.txt")

    s2= txt1.getVob()
    txt2 = Text()
    #txt.readConll2005Sentences("/Users/quynhdo/Documents/WORKING/Data/CoNLL2005/conll05st-release/test.brown/test.brown.txt")
    txt2.readConll2009Sentences("/Users/quynhdo/Documents/WORKING/MYWORK/EACL/CoNLL2009-ST-English2/CoNLL2009-ST-English-evaluation.txt")

    s3= txt2.getVob()

    s = s1.union(s2)
    s= s.union(s3)

    f = open("/Users/quynhdo/Downloads/allWords.txt", "r")
    vobn = set()
    for line in f.readlines():
        line=line.strip()
        vobn.add(line)

    print (len(vobn))
    print (len(vobn.intersection(s)))

    print (len(vobn.intersection(s))/len(s))

    print (len(vobn.intersection(s))/len(vobn))

    print (len(s))