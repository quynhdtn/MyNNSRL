import pycrfsuite
from liir.nlp.nnsrl.features import Feature
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

__author__ = 'quynhdo'

class DataSet(list):

    def __init__(self):
        list.__init__(self)

    def add(self, ins, lbl = None):
        self.append((ins, lbl))

    def addSeq(self, ins):
        if isinstance(ins, list):
            self.append(ins)


    def extractFeature(self,f):
        if isinstance(f,list):
                for ff in f:
                    for x in self:
                        if isinstance(x, tuple):
                            x[0].extractFeature(ff)
                        if isinstance(x, list):
                            for xi in x:
                                xi[0].extractFeature(ff)
        else:
            for x in self:
                if isinstance(x, tuple):
                     x[0].extractFeature(f)
                if isinstance(x, list):
                            for xi in x:
                                xi[0].extractFeature(f)




    # this only works for non-sequence classification
    def asMatrix(self, fl, use_sparse = False):
        s = 0
        for f in fl:
            s += f.size()

        rs = lil_matrix((len(self),s), dtype=float)

        offset = 0
        for f in fl:
                for i in range(len(self)):
                    v = f.getRepresentationIndex(self[i][0].feature_cache[f.feature_name], offset)
                    if v != -1:
                        if isinstance(v, int):
                            rs[i, v]=1
                        if isinstance(v, list):
                            for vv in v:
                                rs[i,vv]=1
                        if isinstance(v, dict):
                            for vv, vk in v.items():
                                rs[i, vv]=vk
                offset += f.size()

        X= None
        if not use_sparse:
            X= rs.todense()
        else:
            X= rs.tocsr()

        Y= [x[1] for x in self]

        return X,Y



    def asSequence(self, fl):
        X=[]
        Y=[]
        for x in self:
            if not isinstance(x, list):
                return None

            sq_dt=[]
            sq_lbl=[]

            for xsq in x:
                sq_dt.append(xsq[0].getIndices2(fl))
                sq_lbl.append(xsq[1])

            iq= pycrfsuite.ItemSequence(sq_dt)

            X.append(iq)
            Y.append(sq_lbl)
        return X,Y


