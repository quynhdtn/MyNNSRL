__author__ = 'quynhdo'
from enum import Enum

class FeatureName(Enum):
    Word = 'Word'
    POS = 'POS'
    InContext = 'InContext'
    LeftChildWord = 'LeftChildWord'
    RightChildWord = 'RightChildWord'
    PredicateWord = 'PredicateWord'
    WEWord = 'WEWord'
    PredicateContext = 'PredicateContext'
    WEPredicateContext = 'WEPredicateContext'
    WEPredicateWord = 'WEPredicateWord'

class WordData(Enum):
    Word = 'word'
    Pos = 'pos'
    Lemma = 'lemma'
    Deprel = 'deprel'


class TargetWord(Enum):
    Word = 'word'
    LeftDep = 'leftChild'
    RightDep = 'rightChild'

class WordPairData(Enum):
    Position = 'position'

