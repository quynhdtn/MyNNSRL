from liir.nlp.nnsrl.features.EnumDeclaration import WordPairData
from liir.nlp.nnsrl.features.Feature import Feature

__author__ = 'quynhdo'

class WordPairContextFeature(Feature):

    def __init__(self, fn,  pos_1 = 0, pos_2 = 1, windows= 3):
        Feature.__init__(self,fn)
        self.pos1=pos_1
        self.pos2=pos_2
        self.windows = int(windows)

    def getFeatureValue(self, ins, used_for_training=True):
        w1 = ins[self.pos1]
        w2 = ins[self.pos2]

        posInSentence1 = w1.sentence.index(w1)
        posInSentence2 = w2.sentence.index(w2)
        v = 0
        if abs(posInSentence1-posInSentence2) <= (self.windows -1)//2:
            v = 1

        if used_for_training:
            self.addFeatureValueToMap(v)
        return v

class WordPairFeature(Feature):
    def __init__(self, fn, wpd, pos_1 = 0, pos_2 = 1):
        Feature.__init__(self,fn)
        self.pos1=pos_1
        self.pos2=pos_2
        self.wpd = wpd

    def getFeatureValue(self, ins, used_for_training=True):
        w1 = ins[self.pos1]
        w2 = ins[self.pos2]

        v = None

        if self.wpd == WordPairData.Position:
            posInSentence1 = w1.sentence.index(w1)
            posInSentence2 = w2.sentence.index(w2)
            if posInSentence1 == posInSentence2:
                v = 0
            if posInSentence1 > posInSentence2:
                v = 1
            if posInSentence1 < posInSentence2:
                v = 2



        if used_for_training:
            self.addFeatureValueToMap(v)
        return v






