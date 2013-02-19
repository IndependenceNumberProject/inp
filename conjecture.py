import sys
sys.path.append(".") # Needed to pass Sage's automated testing

from sage.all import *
from inp import INPGraph
import itertools
import operator

class GraphBrain(SageObject):
    _default_graph_invariants = [Graph.diameter, Graph.radius]
    _default_unary_operators = [sqrt]
    _default_binary_commutative_operators = [operator.add]
    _default_binary_noncommutative_operators = [operator.sub]

    def __init__(self, name=None, comparator=None, target=None, graphs=[], complexity=1,
                 graph_invariants=_default_graph_invariants,
                 unary_operators=_default_unary_operators,
                 binary_commutative_operators=_default_binary_commutative_operators,
                 binary_noncommutative_operators=_default_binary_noncommutative_operators):
        self.name = name
        self.comparator = comparator
        self.target = target
        self.graphs = graphs
        self.complexity = complexity
        self.graph_invariants = graph_invariants
        self.unary_operators = unary_operators
        self.binary_commutative_operators = binary_commutative_operators
        self.binary_noncommutative_operators = binary_noncommutative_operators

    def conjecture(self):
        pass

    def expressions(self, complexity, _cache=None):
        r"""
        Return all possible expressions of the given complexity.

        EXAMPLES:

        ::
            sage: brain = GraphBrain()
            sage: brain.graph_invariants = [Graph.diameter, Graph.radius, Graph.order]
            sage: brain.unary_operators = [sqrt]
            sage: brain.binary_commutative_operators = [operator.add]
            sage: brain.binary_noncommutative_operators = [operator.sub]
            sage: brain.target = Graph.diameter
            sage: brain.expressions(1)
            [radius(G), order(G)]
            sage: brain.target = Graph.order
            sage: brain.expressions(1)
            [diameter(G), radius(G)]
            sage: brain.expressions(2)
            [sqrt(diameter(G)), sqrt(radius(G))]
            sage: brain.expressions(3)
            [diameter(G)^(1/4), radius(G)^(1/4), 0, 2*diameter(G), radius(G) - diameter(G), radius(G) + diameter(G), -radius(G) + diameter(G), radius(G) + diameter(G), 0, 2*radius(G)]
        """
        if _cache is None:
            _cache = {}

        if complexity not in _cache:
            if complexity < 1:
                return []
            elif complexity == 1:
                _cache[1] = [GraphExpression(self, [inv]) for inv in self.graph_invariants if inv != self.target]
            else:
                _cache[complexity] = []

                # Unary operators
                for expr in self.expressions(complexity - 1, _cache):
                    for op in self.unary_operators:
                        _cache[complexity].append(expr.operate(op))

                # Binary operators
                for k in range(1, complexity - 1):
                    for a in self.expressions(k, _cache):
                        for b in self.expressions(complexity - 1 - k, _cache):
                            # Noncommutative
                            for op in self.binary_noncommutative_operators:
                                _cache[complexity].append(a.operate(op, b))

                            # Commutative
                            if k <= complexity - 1 - k:
                                for op in self.binary_commutative_operators:
                                    _cache[complexity].append(a.operate(op, b))

        return _cache[complexity]
        
class GraphExpression(SageObject):

    def __init__(self, brain, rpn_stack):
        """Constructs a new GraphExpression from the given stack of functions."""
        self.brain = brain
        self.rpn_stack = rpn_stack
        super(GraphExpression, self).__init__()

    # def __eq__(self, other):
    #     return self.expression() == other.expression()

    def _repr_(self):
        return repr(self.expression())

    def _latex_(self):
        return latex(self.expression())

    def copy(self):
        return GraphExpression(self.brain, self.rpn_stack[:])

    def operate(self, op, expr=None):
        copy = self.copy()
        if expr is not None: copy.extend(expr.rpn_stack[:])
        copy.append(op)
        return copy

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
