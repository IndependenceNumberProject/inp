import sys
sys.path.append(".") # Needed to pass automated testing.

import json
import collections
import itertools
from functools import wraps
from sage.all import *

import operator
import math
import inp

class GraphBrain(SageObject):
    _default_graph_invariants = [Graph.average_distance, Graph.diameter, Graph.girth,
                                 inp.INPGraph.matching_number, Graph.order, Graph.size,
                                 Graph.wiener_index, Graph.szeged_index, inp.INPGraph.residue,
                                 inp.INPGraph.fractional_alpha, inp.INPGraph.annihilation_number,
                                 inp.INPGraph.min_degree, inp.INPGraph.max_degree,
                                 Graph.average_degree]
    _default_unary_operators = [sqrt]
    _default_binary_commutative_operators = [operator.add, operator.mul]
    _default_binary_noncommutative_operators = [operator.sub, operator.truediv, operator.pow]

    def __init__(self, name, graphs, target, complexity=1,
                 graph_invariants=_default_graph_invariants,
                 unary_operators=_default_unary_operators,
                 binary_commutative_operators=_default_binary_commutative_operators,
                 binary_noncommutative_operators=_default_binary_noncommutative_operators,
                 ):
        self.name = name
        self.graphs = graphs
        self.target = target
        self.graph_invariants = graph_invariants
        self.unary_operators = unary_operators
        self.binary_commutative_operators = binary_commutative_operators
        self.binary_noncommutative_operators = binary_noncommutative_operators
        self.complexity = complexity
        self.conjectures = self._make_conjectures(self.complexity)

    def _repr_(self):
        return "Graph brain at complexity {0} (contains {1} graphs, {2} conjectures)".format(self.complexity, len(self.graphs), len(self.conjectures))

    def _latex_(self):
        return latex(repr(self))

    def _make_conjectures(self, complexity):
        if self.complexity < 1:
            return GraphExpression([])
        elif self.complexity == 1:
            return [GraphExpression[f] for f in self.graph_invariants if not f == self.target]
        else:
            new_expressions = []

            for s in self._make_conjectures(complexity - 1):
                for op in self.unary_operators:
                    new_expressions += [s.append(op)]

            for i in range(1, complexity - 1):
                for lhs in self._make_conjectures(i):
                    for rhs in self._make_conjectures(complexity - 1 - i):
                        for op in self.binary_noncommutative_operators:
                            new_expressions += [lhs.extend(rhs).append(op)]

            for k in range(1, ceil(float(self.complexity)/2)):
                
                if k == complexity - 1 - k:
                    for lhs, rhs in itertools.combinations_with_replacement(self._make_conjectures(k)):
                        for op in self.binary_commutative_operators:
                            new_expressions += [lhs.extend(rhs).append(op)]

                else:



class GraphStatement(SageObject):
    def __init__(self): 
        pass

    def _repr_(self):
        pass

    def _latex_(self):
        pass


class GraphExpression(SageObject):
    def __init__(self, rpn_stack, brain):
        self.rpn_stack = rpn_stack
        self.brain = brain
        
    def _repr_(self):
        return repr(self.expression())

    def _latex_(self):
        return latex(self.expression())

    def expression(self):
        g = var('G')
        stack = []
        for op in self.rpn_stack:
            if op in self.brain.graph_invariants:
                func = function(op.__name__, nargs=1, evalf_func=op)
                stack.append(func(g))
            elif op in self.brain.unary_operators:
                stack.append(op(stack.pop()))
            elif op in self.brain.binary_commutative_operators or op in self.brain.binary_noncommutative_operators:
                stack.append(op(stack.pop(), stack.pop()))
        return stack.pop()