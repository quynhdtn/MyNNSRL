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

    def getFeatureValue(self, ins):
        return self.val.getFeatureValue(ins)


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



    def extractFeature(self, ins, indices=None, offset=0):
        v = self.getFeatureValue(ins)
        if  v is None:
            return np.zeros(self.wedict.we_size)
        if isinstance(v, list) or isinstance(v, set):
            arr = []
            for vv in v:
                av = self.wedict.getWE(vv)
                arr.append(av)

            arr = np.asmatrix(arr)


            arr = np.mean(arr, axis=0)

        else:

            arr = self.wedict.getWE(v)

        return arr


