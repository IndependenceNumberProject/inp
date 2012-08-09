import abc

class UpperBoundBase(object):
    @staticmethod
    @abc.abstractmethod
    def bound(g):
        return

class MatchingUpperBound(UpperBoundBase):
    @staticmethod
    def bound(g):
        return g.num_verts() - int(g.matching(value_only=True))