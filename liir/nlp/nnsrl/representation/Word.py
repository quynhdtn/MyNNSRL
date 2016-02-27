class Word(object):

        def __init__(self,  idx=None,form=None, useDependency=True, useConstituent=False):
            self.Id = idx
            self.Form = form
            self.Word = None # lower case of Form
            self.Lemma = None
            self.Pos= None
            self.useDependency = useDependency
            self.useConstituent = useConstituent
            if useDependency:
                self.Head = None
                self.Deprel = None
                self.Children = []
            if useConstituent:
                self.Parsebit = None
            self.Sentence=None


class Predicate(Word):
    def __init__(self):
        self.Sense = None
        if self.useDependency:
            self.Arguments = {}
        if self.useConstituent:
            self.Arguments = []

    def clear(self):
        self.Sense = None
        if isinstance(self.Arguments , dict):
            self.Arguments = {}
        else:
            if isinstance(self.Arguments , list):
                self.Arguments = []



