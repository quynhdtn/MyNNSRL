from liir.nlp.nnsrl.representation.Word import Word, Predicate

__author__ = 'quynhdo'

import re


# this class define a sentence which is a list of Word
class Sentence(list):
    def __init__(self):   # value should be  a list of Word
        list.__init__(self)

    def getPredicates(self):
        return [w for w in self if isinstance(w, Predicate)]

# Sentence in Conll 2009
class SentenceConll2009(Sentence):
    def __init__(self, data_lines=None):
        Sentence.__init__(self)
        if data_lines is None:
            return
        if not isinstance(data_lines, list):
            return
        pred_id = 0
        dt = []
        for line in data_lines:
            temps=re.split("\\s+", line)
            dt.append(temps)
            w = Word(int(temps[0])-1, temps[1])
            w.word = temps[1].lower()
            w.lemma = temps[3]
            w.pos = temps[5]
            w.head = int(temps[9])-1
            w.deprel = temps[11]

            if "Y" in set(temps[12]):
                w.__class__ = Predicate
                w.sense = temps[13]

            self.append(w)

        # read srl information
        for pred in self:
            if isinstance(pred, Predicate):
                args={}
                for j in range(len(data_lines)):
                    tmps = dt[j]
                    lbl = tmps[14+pred_id]
                    if lbl != "_":
                        args[self[int(tmps[0])-1]]=lbl

                pred.arguments = args
                pred_id += 1

        for w in self:
            w.sentence = self

        self.doDependency()

    def doDependency(self):
        for w in self:
            self[w.head].children.append(w)
            w.head = self[w.head]



class SentenceConll2005(Sentence):  # Sentence in Conll 2005

    def __init__(self, data_lines=None):
        Sentence.__init__(self)
        if data_lines is None:
            return
        if not isinstance(data_lines, list):
            return
        pred_id = 0
        idx=0
        dt = []
        for line in data_lines:
            temps=re.split("\\s+", line)
            # print (temps)
            dt.append(temps)
            w = Word(idx, temps[0], False, True)
            pos = temps[1]

            if pos == "(":
                pos= "-lrb-"
            if pos == ")":
                pos = "-rrb-"
            w.pos = pos
            w.parsebit = temps[2]
            w.word = temps[0].lower()

            if temps[5] != "-":
                w.__class__= Predicate
                w.Sense = temps[5]+"."+temps[4]
            self.append(w)
            idx += 1

        # read srl information
        for pred in self:
            if isinstance(pred, Predicate):
                args=[]
                j = 0
                while j < len(data_lines):
                    tmps = dt[j]
                    lbl = tmps[6+pred_id]
                    match = re.match('\((.+)\*\)', lbl)
                    if match:
                        args.append("B-"+match.group(1))
                        j += 1
                    else:
                        match = re.match('\((.+)\*', lbl)
                        if match:
                            args.append("B-"+match.group(1))
                            for k in range(j+1, len(data_lines)):
                                l1 = data_lines[k]
                                tmps1 = re.split("\\s+",l1)
                                match1 = re.match('\*\)', tmps1[6+pred_id])
                                args.append("I-"+match.group(1))
                                if match1:
                                    j = k+1
                                    break
                        else:
                            args.append("O")
                            j += 1

                pred.arguments = args
                pred_id += 1
        for w in self:
            w.sentence = self


