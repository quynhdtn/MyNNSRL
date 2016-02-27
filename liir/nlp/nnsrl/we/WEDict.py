__author__ = 'quynhdo'

import numpy as np


class WEDict:

    def __init__(self, full_dict_path):
        f = open(full_dict_path, "r")
        self.full_dict = {}
        self.we_size = -1
        for l in f.readlines(): # read the full dictionary
            l = l.strip()
            tmps = l.split("\\s+")
            if len(tmps) > 1:
                we = []
                if self.we_size == -1:
                    self.we_size = len(tmps)-1
                for i in range(1, len(tmps)):
                    we.append(float(tmps[i].strip()))

                self.full_dict[tmps[0]]= np.asarray(we)

        f.close()

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
        for w in vob:
            if w in self.full_dict.keys():
                f.write(w)
                f.write(" ")
                we = self.full_dict[w]
                for val in we:
                    f.write(str(val))
                    f.write(" ")
                f.write("\n")
        f.close()
