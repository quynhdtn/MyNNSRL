from liir.nlp.nnsrl.representation.Text import Text
from liir.nlp.nnsrl.we.WEDict import WEDict

__author__ = 'quynhdo'
# this file is used to extract the WE for words in the dataset to avoid processing with very large WE dict many times
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conll2009", "--conll2009", action="store_true")
    parser.add_argument("-conll2005", "--conll2005", action="store_true")
    parser.add_argument("srldata", help="SRL Data path")
    parser.add_argument("wedata", help="WE Data path")
    parser.add_argument("output", help="Output")
    args = parser.parse_args()
    txt = Text()

    if args.conll2009:
        txt.readConll2009Sentences(args.srldata)
    if args.conll2005:
        txt.readConll2005Sentences(args.srldata)

    d = WEDict(args.wedata)
    d.extractWEForVob(txt.getVob(), args.output)



