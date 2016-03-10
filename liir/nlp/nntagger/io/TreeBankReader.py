from liir.nlp.nnsrl.representation.Sentence import Sentence
from liir.nlp.nnsrl.representation.Text import Text
import nltk
from liir.nlp.nnsrl.representation.Word import Word

__author__ = 'quynhdo'

def readTreeBank():
    txt=Text()
    tb = nltk.corpus.treebank.tagged_sents()
    for sen in tb:
        s= Sentence()
        for w in sen:
            mw = Word(useConstituent=False, useDependency=False)
            mw.form = w[0]
            mw.id = sen.index(w)
            mw.word = w[0].lower()
            mw.pos= w[1]
            s.append(mw)
            mw.sentence=s


        txt.append(s)

    return txt


def readBrown():
    txt=Text()
    tb = nltk.corpus.brown.tagged_sents()
    for sen in tb:
        s= Sentence()
        for w in sen:
            mw = Word(useConstituent=False, useDependency=False)
            mw.id = sen.index(w)
            mw.form = w[0]
            mw.word = w[0].lower()
            mw.pos= w[1]
            s.append(mw)
            mw.sentence=s

        txt.append(s)


    return txt


