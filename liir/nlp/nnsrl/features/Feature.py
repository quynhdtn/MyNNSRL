__author__ = 'quynhdo'


class Feature(object):

    def __init__(self, fn):
        '''
        :param fn: Feature Name
        :return:
        '''

        self.map={} # mapping from feature string - index
        self.current_index = 0
        self.feature_name = fn

    def addFeatureValueToMap(self, s):
        '''
        add a feature string to the map
        :param s:
        :return:
        '''
        if isinstance(s, list) or isinstance(s, set):
            for v in s :
                self.addFeatureValueToMap(v)
        else:
            if not s in self.map.keys():
                self.map[s] = self.current_index
                self.current_index += 1

    def size(self):
        return self.current_index

    def getFeatureValue(self, ins, used_for_training=True):
        pass

    def getIndexSingle(self, value, offset):

        if value in self.map.keys():
            return self.map[value] + offset

        return -1

    def getRepresentationIndex(self, value, offset=0):
        if isinstance(value, list) or isinstance(value, set):
            vals = set()
            for v in value:
                vals.add(self.getIndexSingle(v, offset))
            return vals

        return self.getIndexSingle(value, offset)
