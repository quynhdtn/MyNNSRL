from liir.nlp.nnsrl.features.WEWrapper import WEWrapper

__author__ = 'quynhdo'

class Instance(object):

    def __init__(self,  val, used_for_training = True):
        self.val = val
        self.feature_cache = {}
        self.used_for_training = used_for_training
        self.indices={}

    #def extractFeature(self, f):
    #    v = f.getFeatureValue(self.val, self.used_for_training)
    #    self.feature_cache[f.feature_name] = v

    def extractFeature(self, f, offset=0):
        if isinstance(f, WEWrapper):
            return f.extractFeature(self.val)
        else:
            f.extractFeature(self.val, indices=self.indices, offset=offset, used_for_training=self.used_for_training)



    def getIndices(self, fl):
        indices = {}

        offset = 0
        for f in fl:
                    v = f.getRepresentationIndex(self.feature_cache[f.feature_name], offset)
                    if v != -1:
                        if isinstance(v, int):
                            indices[v]=1
                        if isinstance(v, list):
                            for vv in v:
                                indices[vv]=1
                        if isinstance(v, dict):
                            for vv, vk in v.items():
                                indices[vv]=vk
                    offset += f.size()

        return indices

    def getIndices2(self, fl):
        indices = {}

        offset = 0
        for f in fl:
                    v = f.getRepresentationIndex(self.feature_cache[f.feature_name], offset)
                    if v != -1:
                        if isinstance(v, int):
                            indices[str(v)]=1
                        if isinstance(v, list):
                            for vv in v:
                                indices[str(vv)]=1
                        if isinstance(v, dict):
                            for vv, vk in v.items():
                                indices[str(vv)]=vk
                    offset += f.size()

        return indices