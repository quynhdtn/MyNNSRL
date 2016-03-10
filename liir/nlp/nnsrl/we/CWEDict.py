import re
from liir.nlp.nnsrl.we.WEDict import WEDict

__author__ = 'quynhdo'
import numpy as np

class CWEDict(WEDict):

    def __init__(self, full_dict_path, full_dict_path2):
        WEDict.__init__(self,full_dict_path)
        f = open(full_dict_path2, "r")
        full_dict2 = {}
        we_size2 = -1
        for l in f.readlines(): # read the full dictionary
            l = l.strip()
            tmps = re.split('\s+', l)
            if len(tmps) > 1:
                we = []
                if we_size2 == -1:
                    we_size2 = len(tmps)-1
                for i in range(1, len(tmps)):
                    we.append(float(tmps[i].strip()))

                full_dict2[tmps[0]]= np.asarray(we)

        f.close()

        for k in self.full_dict.keys():
            arr1 = self.full_dict[k]
            arr2 = full_dict2[k]
            arr= np.concatenate((arr1,arr2))
            self.full_dict[k]=arr

        self.we_size += we_size2

    def getFullVobWE(self):
        return np.asarray([v for v in self.full_dict.values()])

    def getFullVobWEAndKeys(self):
        k = []
        t = []
        for item in self.full_dict.items():
            k.append(item[0])
            t.append(item[1])

        return np.asarray(k), np.asarray(t)

    def getWE(self, w):
        we = None
        if w in self.full_dict.keys():
            we = self.full_dict[w]
        else:
            we = np.zeros(self.we_size)
        return we

    def extractWEForVob(self, vob, output):
        f = open(output, "w")
        c = 0
        for w in vob:
            if w in self.full_dict.keys():
                f.write(w)
                f.write(" ")
                we = self.full_dict[w]
                c += 1
                for val in we:
                    f.write(str(val))
                    f.write(" ")
                f.write("\n")
        f.close()
        print ( "Words in WE dict: ")
        print (str(c) + "/" + str(len(vob)))
