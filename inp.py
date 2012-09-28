r"""
Search for difficult graphs in the Independence Number Project.

AUTHORS:

- Patrick Gaskill (2012-09-16): v0.2 refactored into INPGraph class

- Patrick Gaskill (2012-08-21): v0.1 initial version
"""

#*****************************************************************************
#       Copyright (C) 2012 Patrick Gaskill <gaskillpw@vcu.edu>
#       Copyright (C) 2012 Craig Larson <clarson@vcu.edu>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************

#import cvxopt.base
##import cvxopt.solvers
#import datetime
#from string import Template
#import subprocess
#import sys
#import time

import sage.all
from sage.all import Graph, graphs

try:
    from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
                            FileTransferSpeed, FormatLabel, Percentage, \
                            ProgressBar, ReverseBar, RotatingMarker, \
                            SimpleProgress, Timer
    has_progressbar = True
except ImportError:
    has_progressbar = False

class INPGraph(Graph):
    __lower_bounds = []
    __upper_bounds = []
    __alpha_properties = []

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)

    @classmethod
    def next_difficult_graph(cls, order=None, verbose=True, write_to_pdf=False):
        # TODO: Is it possible to write good tests for this?
        # TODO: Write this function including a progress bar
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=300).start()
        for i in range(300):
            #time.sleep(0.01)
            pbar.update(i+1)
        pbar.finish()

    def is_difficult(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function determines if the graph is difficult as described by
        INP theory.

        NOTES:

        The return value of this function may change depending on the functions
        included in the __lower_bounds, __upper_bounds, and __alpha_properties
        settings.
        """
        if self.has_alpha_property():
            return False

        lbound = ceil(self.best_lower_bound())
        ubound = floor(self.best_upper_bound())

        if lbound == ubound:
            return False

        return True

    def best_lower_bound(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function computes a lower bound for the independence number of the
        graph.

        NOTES:

        The return value of this function may change depending on the functions
        included in the __lower_bounds setting.
        """
        # The default bound is 1
        lbound = 1

        for func in self.__lower_bounds:
            try:
                new_bound = self.func()
                if new_bound > lbound:
                    lbound = new_bound
            except ValueError:
                pass

        return lbound

    def best_upper_bound(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function computes an upper bound for the independence number of
        the graph.

        NOTES:

        The return value of this function may change depending on the functions
        included in the __upper_bounds setting.
        """
        # The default upper bound is the number of vertices
        ubound = g.order()

        for func in self.__upper_bounds:
            try:
                new_bound = self.func()
                if new_bound < ubound:
                    ubound = new_bound
            except ValueError:
                pass

        return ubound

    def has_alpha_property(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function determines if the graph satisifes any of the known
        alpha-properties or alpha-reductions.

        NOTES:

        The return value of this function may change depending on the functions
        included in the __alpha_properties setting.
        """
        for func in self.__alpha_properties:
            try:
                if self.func():
                    return True
            except ValueError:
                pass

        return False

    def write_to_pdf(self):
        # TODO: Write this function
        # TODO: Is it possible to write good tests for this?
        # TODO: The latex template should be written using .sty files
        pass

    def matching_number(self):
        # TODO: This needs to be updated when Sage 5.3 is released.
        r"""
        Compute the traditional matching number `\mu`.

        EXAMPLES:

        ::
            sage: tests = [2, graphs.CompleteGraph(3), graphs.PathGraph(3), \
                           graphs.StarGraph(3), 'EXCO', graphs.CycleGraph(5), \
                           graphs.PetersenGraph()]
            sage: [INPGraph(g).matching_number() for g in tests]
            [0, 1, 1, 1, 2, 2, 5]

        WARNINGS:

        Ideally we would set use_edge_labels=False to ignore edge weighting and
        always compute the traditional matching number, but there is a bug
        in Sage 5.2 that returns double this number. Calling this on an
        edge-weighted graph will NOT give the usual matching number.
        """
        return int(self.matching(value_only=True))

    mu = matching_number

    def independence_number(self):
        r"""
        Compute the independence number using the Sage built-in independent_set
        method. This uses the Cliquer algorithm, which does not run in
        polynomial time.

        EXAMPLES:

        ::
            sage: tests = [2, graphs.CompleteGraph(3), graphs.PathGraph(3), \
                           graphs.StarGraph(3), 'EXCO', graphs.CycleGraph(5), \
                           graphs.PetersenGraph()]
            sage: [INPGraph(g).independence_number() for g in tests]
            [2, 1, 2, 3, 4, 2, 4]

        """
        return int(len(self.independent_set()))

    alpha = independence_number

    def bipartite_double_cover(self):
        r"""
        Return a bipartite double cover of the graph, also known as the
        bidouble.

        EXAMPLES:

        ::
            sage: b = INPGraph(2).bipartite_double_cover()
            sage: b.is_isomorphic(Graph(4))
            True

        ::
            sage: b = INPGraph(graphs.CompleteGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(graphs.CycleGraph(6))
            True

        ::
            sage: b = INPGraph(graphs.PathGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('EgCG'))
            True

        ::
            sage: b = INPGraph(graphs.StarGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('Gs?GGG'))
            True

        ::
            sage: b = INPGraph('EXCO').bipartite_double_cover()
            sage: b.is_isomorphic(Graph('KXCO?C@??A_@'))
            True

        ::
            sage: b = INPGraph(graphs.CycleGraph(5)).bipartite_double_cover()
            sage: b.is_isomorphic(graphs.CycleGraph(10))
            True

        ::
            sage: b = INPGraph(graphs.PetersenGraph()).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('SKC_GP@_a?O?C?G??__OO?POAI??a_@D?'))
            True
        """
        return INPGraph(self.tensor_product(graphs.CompleteGraph(2)))

    bidouble = bipartite_double_cover
    kronecker_double_cover = bipartite_double_cover
    canonical_double_cover = bipartite_double_cover

    def closed_neighborhood(self, verts):
        # TODO: Write tests
        if isinstance(verts, list):
            neighborhood = []
            for v in verts:
                neighborhood += [v] + self.neighbors(v)
            return list(set(neighborhood))
        else:
            return [verts] + self.neighbors(verts)

    def open_neighborhood(self, verts):
        # TODO: Write tests
        if isinstance(verts, list):
            neighborhood = []
            for v in verts:
                neighborhood += self.neighbors(v)
            return list(set(neighborhood))
        else:
            return self.neighbors(verts)

    def max_degree(self):
        # TODO: Write tests
        return max(self.degree())

    def min_degree(self):
        # TODO: Write tests
        return min(self.degree())

    def union_MCIS(self):
        # TODO: Write more tests
        r"""
        Return a union of maximum critical independent sets (MCIS).

        EXAMPLES:

        ::
            sage: G = INPGraph('Cx')
            sage: G.union_MCIS()
            [0, 1, 3]

        ::
            sage: G = INPGraph(graphs.CycleGraph(4))
            sage: G.union_MCIS()
            [0, 1, 2, 3]

        """
        b = self.bipartite_double_cover()
        alpha = b.order() - b.matching_number()

        result = []

        for v in self.vertices():
            test = b.copy()
            test.delete_vertices(b.closed_neighborhood([(v,0), (v,1)]))
            alpha_test = test.order() - test.matching_number() + 2
            if alpha_test == alpha:
                result.append(v)

        return result

    def has_max_degree_order_minus_one(self):
        # TODO: Write tests
        return self.max_degree() == self.order() - 1
    has_max_degree_order_minus_one.__is_alpha_property = True

    def is_claw_free(self):
        # TODO: Write tests
        subsets = combinations_iterator(self.vertices(), 4)
        for subset in subsets:
            if self.subgraph(subset).degree_sequence() == [3,1,1,1]:
                return False
        return True
    is_claw_free.__is_alpha_property = True

    def has_pendant_vertex(self):
        return 1 in self.degree()
    has_pendant_vertex.__is_alpha_property = True

    def has_simplicial_vertex(self):
        # TODO: Write tests
        for v in self.vertices():
            if self.subgraph(g.neighbors(v)).is_clique():
                return True

        return False
    has_simplicial_vertex.__is_alpha_property = True

    def is_KE(self):
        # TODO: Write tests
        c = self.union_MCIS()
        nc = []
        for v in c:
            nc.extend(g.neighbors(v))

        return list(set(c + nc)) == self.vertices()
    is_KE.__is_alpha_property = True

    def is_almost_KE(self):
        # TODO: Write tests
        subsets = combinations_iterator(self.vertices(), self.order() - 1)
        for subset in subsets:
            if self.subgraph(subset).is_KE():
                return True

        return False
    is_almost_KE.__is_alpha_property = True

    def has_nonempty_KE_part(self):
        # TODO: Write tests
        if self.union_MCIS():
            return True
        else:
            return False
    has_nonempty_KE_part.__is_alpha_property = True

    def is_foldable(self):
        # TODO: Write tests
        # TODO: Write this function
        pass
    is_foldable.__is_alpha_property = True

    def residue(self):
        # TODO: Write tests
        seq = self.degree_sequence()

        while seq[0] > 0:
            d = seq.pop(0)
            seq[:d] = [k-1 for k in seq[:d]]
            seq.sort(reverse=True)

        return len(seq)
    residue.__is_lower_bound = True

    def average_degree_bound(self):
        # TODO: Write tests
        n = Integer(self.order())
        d = Rational(self.average_degree())
        return n / (1 + d)
    average_degree_bound.__is_lower_bound = True

    # TODO: Include more functions from survey
    # TODO: Include functions from old code
    