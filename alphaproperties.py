import abc

class AlphaPropertyBase(object):
    @staticmethod
    @abc.abstractmethod
    def has_property(g):
        return

class MaxDegreeProperty(AlphaPropertyBase):
    @staticmethod
    def has_property(g):
        max_degree = g.degree_sequence()[0]
        if max_degree == g.num_verts() - 1:
            return True
        else:
            return False