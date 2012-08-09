import abc

class LowerBoundBase(object):
    @staticmethod
    @abc.abstractmethod
    def bound(g):
        return

class MatchingLowerBound(LowerBoundBase):
    @staticmethod
    def bound(g):
        return g.num_verts() - 2 * int(g.matching(value_only=True))