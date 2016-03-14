import pycrfsuite
from sklearn import preprocessing
from liir.nlp.nnsrl.features import Feature
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

__author__ = 'quynhdo'

class DataSet(list):

    def __init__(self):
        list.__init__(self)
        self.label_map={}

    def add(self, ins, lbl = None):
        self.append((ins, lbl))

    def addSeq(self, ins):
        if isinstance(ins, list):
            self.append(ins)

    def longestInstance(self):
        m = 0
        for d in self:
            if len(d[0])>m:
                m=len(d[0])
        return m

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



    def asSequenceNumpy(self, fl, store_label_map = False):
        s = 0
        for f in fl:
            s += f.size()
        X=[]
        Y=[]
        for x in self:
            if not isinstance(x, list):
                return None

            sq_dt= lil_matrix((len(x),s), dtype=float)
            sq_lbl=[]

            for i in range(len(x)):
                xsq =x [i]

                ids= xsq[0].getIndices2(fl)
                for idx, v in ids.items():
                    sq_dt[i, idx]=v

                sq_lbl.append(xsq[1])
            X.append(sq_dt.todense())
            Y.append(sq_lbl)

        if store_label_map:
            Yflat=[]
            for yy in Y:
                for yyy in yy:
                    Yflat.append(yyy)
            classes = list(np.unique(Yflat))
            for c in classes:
                self.label_map[c]=classes.index(c)

        Ybi = []

        for y in Y:
                ybb=[]
                for yy in y:
                    yb = np.zeros((1, len(self.label_map)))
                    yb[0,self.label_map[yy]]=1
                    ybb.append(yb)
                Ybi.append(ybb)
        Y=Ybi

        X_t = np.asarray(X)
        print(X_t.shape)

        return X,Y


