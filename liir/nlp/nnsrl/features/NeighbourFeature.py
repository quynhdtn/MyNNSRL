from liir.nlp.nnsrl.features.EnumDeclaration import WordData
from liir.nlp.nnsrl.features.Feature import Feature
from liir.nlp.nnsrl.representation.Word import Word

__author__ = 'quynhdo'
class NeighbourFeature(Feature):
    def __init__(self, fn, wd, tw, nei, pos = 0):
        '''

        :param fn:  Feature Name
        :param wd: Word Data
        :param tw: Target Word
        :param pos: target position if the instance to extract is not a single object
        :return:
        '''

        Feature.__init__(self,fn)
        self.word_data = wd
        self.target_word = tw
        self.target_position = pos
        self.nei = nei

    def getFeatureValue(self, ins):
        wIns = None
        if isinstance(ins, Word):
            wIns=ins
        else:

            wIns = ins[self.target_position]

        wTarget = wIns.getWord(self.target_word)
        if wTarget is None:
            return None

        if wTarget.id +self.nei >= 0 and wTarget.id +self.nei <len(wTarget.sentence):
            wTarget = wTarget.sentence[wTarget.id +self.nei]
        else:
            return None

        rs = None
        if self.word_data  == WordData.Word:
            rs = wTarget.word

        else :
            if self.word_data == WordData.Pos:
                rs = wTarget.pos

            else:
                if self.word_data == WordData.Deprel:
                    rs = wTarget.deprel
                else:
                    if self.word_data == WordData.Lemma:
                        rs = wTarget.lemma
                    else:
                        if self.word_data == WordData.Capital:
                            rs = wTarget.form[0].isupper()



        return rs
