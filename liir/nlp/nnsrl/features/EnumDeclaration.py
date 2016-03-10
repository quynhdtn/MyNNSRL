__author__ = 'quynhdo'
from enum import Enum

class FeatureName(Enum):
    Word = 'Word'
    POS = 'POS'
    InContext = 'InContext'
    LeftChildWord = 'LeftChildWord'
    WELeftChildWord = 'WELeftChildWord'
    RightChildWord = 'RightChildWord'
    WERightChildWord = 'WERightChildWord'

    PredicateWord = 'PredicateWord'
    WEWord = 'WEWord'
    PredicateContext = 'PredicateContext'
    WEPredicateContext = 'WEPredicateContext'
    WEPredicateWord = 'WEPredicateWord'
    IsCapital = 'IsCapital'
    NeighbourWord = 'NeighbourWord'
    WENeighbourWord = 'WENeighbourWord'

    NeighbourPOS = 'NeighbourPOS'


class WordData(Enum):
    Word = 'word'
    Pos = 'pos'
    Lemma = 'lemma'
    Deprel = 'deprel'
    Capital = 'capital'


class TargetWord(Enum):
    Word = 'word'
    LeftDep = 'leftChild'
    RightDep = 'rightChild'

class WordPairData(Enum):
    Position = 'position'

