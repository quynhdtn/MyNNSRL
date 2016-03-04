from liir.nlp.nnsrl.features.ContextFeature import ContextFeature
from liir.nlp.nnsrl.features.EnumDeclaration import FeatureName
from liir.nlp.nnsrl.features.Feature import Feature
import numpy as np
__author__ = 'quynhdo'

class WEWrapper(Feature):

    def __init__(self, f, wedict):
        Feature.__init__(self, FeatureName("WE" + f.feature_name.value))
        self.val = f
        self.wedict = wedict

    def size(self):
        if isinstance(self.val, ContextFeature):
            return self.wedict.we_size * self.val.windows
        return self.wedict.we_size

    def getFeatureValue(self, ins, used_for_training = True):
        return self.val.getFeatureValue(ins, used_for_training)



    def getRepresentationIndex(self, value, offset=0):
        rs = {}
        p = offset
        if isinstance(value, list):
            for v in value:
                if v in self.wedict.full_dict.keys():
                    ve = self.wedict.full_dict[v]
                    for j in range (self.wedict.we_size):
                        rs[p + j] = ve [j]
                    p += self.wedict.we_size

        else:

            if value in self.wedict.full_dict.keys():
                    ve = self.wedict.full_dict[value]
                    for j in range (self.wedict.we_size):
                        rs[p + j] = ve [j]

        return rs



