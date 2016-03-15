import pycrfsuite
from sklearn import preprocessing
from liir.nlp.nnsrl.features import Feature
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import time
from liir.nlp.nnsrl.features.WEWrapper import WEWrapper

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
            if len(d)>m:
                m=len(d)
        return m


    def extractFeatureForMatrix(self, f, store_label_map=True):
        WEf = []
        NonWEf=[]
        rs=None

        for ff in f:

                if isinstance(ff, WEWrapper):
                    WEf.append(ff)
                else:
                    NonWEf.append(ff)

        print(WEf)
        print(NonWEf)
        if (len(NonWEf) >0):

            offset=  0
            for i in range(0, len(NonWEf)):
                if i>0:
                    offset += NonWEf[i-1].size()
                print ("Extracting for ")
                print (NonWEf[i].feature_name)
                start_time = time.time()
                for x in self:
                    if isinstance(x, tuple):

                        x[0].extractFeature(NonWEf[i], offset=offset)

                    if isinstance(x, list):
                        for xi in x:
                            xi[0].extractFeature(ff, offset=offset)
                print("--- %s seconds ---" % (time.time() - start_time))

            # make matrix:
            s = 0
            for ff in NonWEf:
                s+= ff.size()
            t=0
            if isinstance(self[0], tuple):
                t+= len(self)
            else:
                for x in self:

                        t+=len(x)

            rs = lil_matrix((t,s), dtype=float)
            ins_pos = 0
            for x in self:
                    if isinstance(x, tuple):
                        for idx, v in x[0].indices.items():
                            rs[ins_pos,idx]=1
                        ins_pos+=1


                    if isinstance(x, list):
                        for xi in x:
                            for idx, v in xi[0].indices.items():
                                rs[ins_pos,idx]=1
                            ins_pos+=1

            rs = rs.todense()
            print(rs.shape)
        for f in WEf:
            print (f.feature_name)
            start_time = time.time()
            m=[]
            for x in self:
                    if isinstance(x, tuple):
                        v= x[0].extractFeature(f)
                        m.append(v)


                    if isinstance(x, list):
                        for xi in x:
                            v = xi[0].extractFeature(f)
                            m.append(v)

            print (m[0].shape)
            m=np.asarray(m)
            print(m.shape)
            if len(m.shape)>2:
                m=m.reshape(m.shape[0],m.shape[2])
            if rs is None:
                rs = m
            else:
                rs = np.concatenate((rs,m), axis=1)
            print(rs.shape)
            print("--- %s seconds ---" % (time.time() - start_time))

        print ("Shape")
        print(rs.shape)

        X=[]
        Y=[]
        idx = 0
        if  isinstance(self[0], tuple):
            X=rs
        else:
            for x in self:
                lx=[]
                ly=[]
                for xi in x:
                    lx.append(rs[idx,:])
                    ly.append(xi[1])
                    idx+=1
                lxm=np.asarray(lx)
                lxm=lxm.reshape(lxm.shape[0],lxm.shape[2])
                print(lxm.shape)
                X.append(lxm)
                Y.append(ly)

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

            print(len(X))
            print (X[0].shape)
            print(len(Y))
        return X,Y

    '''
    def extractFeature(self,f):

        print ("Start extracting feature ")

        if isinstance(f,list):
                for ff in f:
                    print (ff.feature_name)
                    start_time = time.time()
                    for x in self:
                        if isinstance(x, tuple):
                            x[0].extractFeature(ff)
                        if isinstance(x, list):
                            for xi in x:
                                xi[0].extractFeature(ff)
                    print("--- %s seconds ---" % (time.time() - start_time))


        else:
            for x in self:
                if isinstance(x, tuple):
                     x[0].extractFeature(f)
                if isinstance(x, list):
                            for xi in x:
                                xi[0].extractFeature(f)


    '''

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
        print ("Start building matrix...")
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


