from liir.nlp.nnsrl.features.EnumDeclaration import WordData
from liir.nlp.nnsrl.features.Feature import Feature

__author__ = 'quynhdo'

class ContextFeature(Feature):
    def __init__(self, fn, wd, tw, pos = 0, windows=3):
        Feature.__init__(self, fn)
        self.word_data = wd
        self.target_word = tw
        self.target_position = pos
        self.windows = int(windows)

    def getFeatureValue(self, ins, used_for_training=True):
        wIns = None
        if self.target_position == -1:
            wIns = ins
        else:
            wIns = ins[self.target_position]

        wTarget = wIns.getWord(self.target_word)
        rs = None
        l = []
        s = wTarget.sentence
        position_in_sentence = s.index(wTarget)
        for i in range(position_in_sentence - (self.windows -1)//2,  position_in_sentence):
            if  i >= 0:
                l.append(s[i])
        l.append(s[position_in_sentence])
        for i in range(position_in_sentence + 1, position_in_sentence + (self.windows -1)//2 +1):
            if i < len(s):
                l.append(s[i])

        rsl = []

        for myw in l:
            rs = None
            if self.word_data  == WordData.Word:
                rs = myw.word

            else :
                if self.word_data == WordData.Pos:
                    rs = myw.pos

                else:
                    if self.word_data == WordData.Deprel:
                        rs = myw.deprel
                    else:
                        if self.word_data == WordData.Lemma:
                            rs = myw.lemma

            rsl.append(rs)

        if used_for_training:
            for r in rsl:
                self.addFeatureValueToMap(r)
        return rsl




