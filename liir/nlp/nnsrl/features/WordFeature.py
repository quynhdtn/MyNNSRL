from liir.nlp.nnsrl.features.EnumDeclaration import WordData
from liir.nlp.nnsrl.features.Feature import Feature
from liir.nlp.nnsrl.representation.Word import Word

__author__ = 'quynhdo'

class WordFeature(Feature):
    def __init__(self, fn, wd, tw, pos = 0):
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

    def getFeatureValue(self, ins, used_for_training=True):
        wIns = None
        if isinstance(ins, Word):
            wIns = ins
        else:


            wIns = ins[self.target_position]

        wTarget = wIns.getWord(self.target_word)
        if wTarget is None:
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


        if used_for_training:
            self.addFeatureValueToMap(rs)
        return rs





