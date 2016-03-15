from liir.nlp.nnsrl.features.ContextFeature import ContextFeature
from liir.nlp.nnsrl.features.EnumDeclaration import FeatureName, WordData, TargetWord
from liir.nlp.nnsrl.features.NeighbourFeature import NeighbourFeature
from liir.nlp.nnsrl.features.WEWrapper import WEWrapper
from liir.nlp.nnsrl.features.WordFeature import WordFeature
from liir.nlp.nnsrl.features.WordPairFeature import WordPairContextFeature
from liir.nlp.nnsrl.we.CWEDict import CWEDict
from liir.nlp.nnsrl.we.WEDict import WEDict

__author__ = 'quynhdo'

class FeatureGenerator(list):
    def __init__(self, feature_file, we_list_file= None):
        list.__init__(self)

        if we_list_file is not None:
            self.we_dicts = {}
            f = open(we_list_file, 'r')
            for line in f.readlines():
                line=line.strip()
                tmps = line.split(" ")
                if len(tmps) == 2:
                    wed = WEDict(tmps[1])
                    self.we_dicts[tmps[0]]=wed
                if len(tmps) == 3:
                    wed = CWEDict(tmps[1], tmps[2])
                    self.we_dicts[tmps[0]]=wed

        self.ParseFeatureFile(feature_file)



    def ParseFeatureFile(self, feature_file):
        f = open(feature_file,  "r")
        for line in f.readlines():
            line=line.strip()
            tmps = line.split(" ")

            if tmps[0]== "WE":
                    attr = {}
                    for j in range(2, len(tmps)):
                        at = tmps[j].split(":")
                        if len(at)==2:
                            attr[at[0]]= at[1]
                    f = self.getFeature(FeatureName(tmps[1]),attr)
                    wf = WEWrapper(f, self.we_dicts[tmps[2]])
                    self.append(wf)

            else:
                    attr = {}
                    for j in range(1, len(tmps)):
                        at = tmps[j].split(":")
                        if len(at)==2:
                            attr[at[0]]= at[1]
                    f = self.getFeature(FeatureName(tmps[0]), attr)
                    self.append(f)



    def getFeature(self, fn, attr = None):
        if fn == FeatureName.Word:
            return WordFeature(fn, WordData.Word, TargetWord.Word)

        if fn == FeatureName.POS:
            return WordFeature(fn, WordData.Pos, TargetWord.Word)

        if fn == FeatureName.PredicateWord:
            return WordFeature(fn, WordData.Word, TargetWord.Word, pos=1)

        if fn == FeatureName.LeftChildWord:
            return WordFeature(fn, WordData.Word, TargetWord.LeftDep)

        if fn == FeatureName.RightChildWord:
            return WordFeature(fn, WordData.Word, TargetWord.RightDep)


        if fn == FeatureName.InContext:
            return WordPairContextFeature(fn, windows=attr['w'])


        if fn == FeatureName.PredicateContext:
            return ContextFeature(fn, WordData.Word, TargetWord.Word, pos=1, windows=attr['w'])

        if fn == FeatureName.NeighbourWord:
            return NeighbourFeature(fn, WordData.Word, TargetWord.Word, nei= int(attr['p']))

        if fn == FeatureName.NeighbourPOS:
            return NeighbourFeature(fn, WordData.Pos, TargetWord.Word, nei= int(attr['p']))

        if fn == FeatureName.PredNeighbourWord:
            return NeighbourFeature(fn, WordData.Word, TargetWord.Word, pos=1, nei= int(attr['p']))


        if fn == FeatureName.PredNeighbourPOS:
            return NeighbourFeature(fn, WordData.Pos, TargetWord.Word, pos=1, nei= int(attr['p']))


        if fn == FeatureName.IsCapital:
            return WordFeature(fn, WordData.Capital, TargetWord.Word)


