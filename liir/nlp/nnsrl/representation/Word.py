from liir.nlp.nnsrl.features.EnumDeclaration import TargetWord


class Word(object):

    def __init__(self,  idx=None,form=None, useDependency=True, useConstituent=False):
            self.id = idx
            self.form = form
            self.word = None # lower case of Form
            self.lemma = None
            self.pos= None
            self.useDependency = useDependency
            self.useConstituent = useConstituent
            if useDependency:
                self.head = None
                self.deprel = None
                self.children = []
            if useConstituent:
                self.parsebit = None
            self.sentence=None

    def getWord(self, tw):
        if tw == TargetWord.Word:
            return self
        if tw == TargetWord.LeftDep:
            if len(self.children) == 0:
                return None
            return self.children[0]

        if tw == TargetWord.RightDep:
            if len(self.children) == 0:
                return None
            return self.children[len(self.children)-1]


class Predicate(Word):
    def __init__(self):
        self.sense = None
        if self.useDependency:
            self.arguments = {}
        if self.useConstituent:
            self.arguments = []

    def clear(self):
        self.sense = None
        if isinstance(self.arguments , dict):
            self.arguments = {}
        else:
            if isinstance(self.arguments , list):
                self.arguments = []



