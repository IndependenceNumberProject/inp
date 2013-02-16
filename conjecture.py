import sys
sys.path.append(".") # Needed to pass Sage's automated testing

from sage.all import *
from inp import INPGraph
#import math
import operator

class GraphBrain(SageObject):
    _default_graph_invariants = [INPGraph.min_degree]
    _default_unary_operators = [sqrt]
    _default_binary_commutative_operators = [operator.add]
    _default_binary_noncommutative_operators = []

    def __init__(self, name=None, target=None, graphs=[], complexity=1,
                 graph_invariants=_default_graph_invariants,
                 unary_operators=_default_unary_operators,
                 binary_commutative_operators=_default_binary_commutative_operators,
                 binary_noncommutative_operators=_default_binary_noncommutative_operators):
        self.name = name
        self.target = target
        self.graphs = graphs
        self.complexity = complexity
        self.graph_invariants = graph_invariants
        self.unary_operators = unary_operators
        self.binary_commutative_operators = binary_commutative_operators
        self.binary_noncommutative_operators = binary_noncommutative_operators

class GraphStatement(SageObject):
    pass

class GraphExpression(SageObject):

    def __init__(self, brain, rpn_stack):
        """Constructs a new GraphExpression from the given stack of functions."""
        self.brain = brain
        self.rpn_stack = rpn_stack
        super(GraphExpression, self).__init__()

    def _repr_(self):
        return repr(self.expression())

    def _latex_(self):
        return latex(self.expression())

    def append(self, x):
        r"""
        Append a command to the right end of the expression stack.

        EXAMPLES:

            ::
                sage: brain = GraphBrain(target=INPGraph.independence_number, graph_invariants=[INPGraph.min_degree, INPGraph.independence_number], unary_operators=[sqrt], binary_commutative_operators=[operator.add], binary_noncommutative_operators=[])
                sage: expr = GraphExpression(brain, [INPGraph.min_degree])
                sage: expr.append(sqrt)
                sage: expr.rpn_stack
                [<unbound method INPGraph.min_degree>, <function sqrt at ...>]
        """
        self.rpn_stack.append(x)

    def extend(self, li):
        r"""
        Extend the expression stack by the given list.

        EXAMPLES:

            ::
                sage: brain = GraphBrain(target=INPGraph.independence_number, graph_invariants=[INPGraph.min_degree, INPGraph.independence_number], unary_operators=[sqrt], binary_commutative_operators=[operator.add], binary_noncommutative_operators=[])
                sage: expr = GraphExpression(brain, [INPGraph.min_degree, sqrt])
                sage: expr.extend([INPGraph.min_degree, operator.add])
                sage: expr.rpn_stack
                [<unbound method INPGraph.min_degree>, <function sqrt at ...>, <unbound method INPGraph.min_degree>, <built-in function add>]
        """
        self.rpn_stack.extend(li)

    def complexity(self):
        r"""
        Return the complexity of the expression, i.e. the length of its stack.

        EXAMPLES:

        ::
            sage: brain = GraphBrain(target=INPGraph.independence_number, graph_invariants=[INPGraph.min_degree, INPGraph.independence_number], unary_operators=[sqrt], binary_commutative_operators=[operator.add], binary_noncommutative_operators=[])
            sage: expr = GraphExpression(brain, [INPGraph.min_degree, sqrt])
            sage: expr.complexity()
            2
        """
        return len(self.rpn_stack)

    def evaluate(self, g):
        r"""
        Evaluate the expression for the given graph.

        EXAMPLES:

        ::
            sage: brain = GraphBrain(target=INPGraph.independence_number, graph_invariants=[INPGraph.min_degree, INPGraph.independence_number], unary_operators=[sqrt], binary_commutative_operators=[operator.add], binary_noncommutative_operators=[])
            sage: expr = GraphExpression(brain, [INPGraph.independence_number])
            sage: g = INPGraph(graphs.PetersenGraph())
            sage: expr.evaluate(g)
            4
            sage: expr = GraphExpression(brain, [INPGraph.min_degree, sqrt])
            sage: expr.evaluate(g)
            sqrt(3)
        """
        stack = []
        for op in self.rpn_stack:
            if op in self.brain.graph_invariants:
                stack.append(op(g))
            elif op in self.brain.unary_operators:
                stack.append(op(stack.pop()))
            elif op in self.brain.binary_commutative_operators + self.brain.binary_noncommutative_operators:
                stack.append(op(stack.pop(), stack.pop()))
        return stack.pop()

    def expression(self, graph_variable='G'):
        r"""
        Return a Sage symbolic expression object.

        EXAMPLES:

        ::
            sage: brain = GraphBrain(target=INPGraph.independence_number, graph_invariants=[INPGraph.min_degree, INPGraph.independence_number], unary_operators=[sqrt], binary_commutative_operators=[operator.add], binary_noncommutative_operators=[])
            sage: expr = GraphExpression(brain, [INPGraph.min_degree, sqrt, INPGraph.min_degree, sqrt, operator.add])
            sage: expr.expression()
            2*sqrt(min_degree(G))
        """
        g = var(graph_variable)
        stack = []
        for op in self.rpn_stack:
            if op in self.brain.graph_invariants:
                func = function(op.__name__, nargs=1, evalf_func=op)
                stack.append(func(g))
            elif op in self.brain.unary_operators:
                stack.append(op(stack.pop()))
            elif op in self.brain.binary_commutative_operators + self.brain.binary_noncommutative_operators:
                stack.append(op(stack.pop(), stack.pop()))
        return stack.pop()
