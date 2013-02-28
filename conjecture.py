import sys
sys.path.append(".") # Needed to pass Sage's automated testing

from sage.all import *
from inp import INPGraph
import itertools
import operator

class GraphBrain(SageObject):
    _default_graph_invariants = [Graph.average_distance, Graph.diameter, Graph.radius, Graph.girth,
                                 INPGraph.matching_number, Graph.order, Graph.size, Graph.szeged_index,
                                 Graph.wiener_index, INPGraph.residue, INPGraph.fractional_alpha,
                                 INPGraph.annihilation_number, INPGraph.lovasz_theta, INPGraph.cvetkovic,
                                 INPGraph.max_degree, INPGraph.min_degree, Graph.average_degree]
    _default_unary_operators = [sqrt]
    _default_binary_commutative_operators = [operator.add, operator.mul]
    _default_binary_noncommutative_operators = [operator.sub, operator.truediv]

    _save_path = os.path.expanduser("~/Dropbox/INP")

    def __init__(self, name=None, comparator=None, target=None, graphs=[], complexity=1,
                 graph_invariants=_default_graph_invariants,
                 unary_operators=_default_unary_operators,
                 binary_commutative_operators=_default_binary_commutative_operators,
                 binary_noncommutative_operators=_default_binary_noncommutative_operators):
        self.name = name
        self.comparator = comparator
        self.target = target

        if not all(isinstance(g, INPGraph) for g in graphs):
            raise TypeError("Graphs must be INPGraph objects.")
        else:
            self.graphs = graphs

        self.complexity = complexity
        self.graph_invariants = graph_invariants
        self.unary_operators = unary_operators
        self.binary_commutative_operators = binary_commutative_operators
        self.binary_noncommutative_operators = binary_noncommutative_operators

    def conjecture(self):
        r"""
        Return a list of true statements that are also significant for at least
        one graph in the brain, that is, the statement gives the tightest bound.
        """
        conjectures = {}
        targets = {id(g): self.target(g) for g in self.graphs}
        #print targets

        while not conjectures:
            
            print "Checking expressions of complexity", self.complexity, "..."

            for expr in self.expressions():
                
                evaluations = {id(g): expr.evaluate(g) for g in self.graphs}
                
                try:
                    # The expression has to be true for all the graphs in the brain.
                    if not all(self.comparator(evaluations[id(g)], targets[id(g)]) for g in self.graphs):
                        # If we're checking <= or >=, we need the expression to have equality for at least one graph.
                        if self.comparator in [operator.le, operator.ge] and all(evaluations[id(g)] != targets[id(g)] for g in self.graphs):
                            continue
                except (TypeError, ValueError) as e:
                    #print "Unable to evaluate", expr, ":", e
                    continue

                for g in self.graphs:
                    gid = id(g)

                    try:
                        evaluation = expr.evaluate(g)
                    except (TypeError, ValueError) as e:
                        #print "Unable to evaluate", expr, "for graph", g.graph6_string(), ":", e
                        break

                    #print "\t", g, evaluation
       
                    if gid in conjectures and evaluation == conjectures[gid]['value']:
                        conjectures[gid]['expressions'].append(expr)
                    elif gid not in conjectures or not self.comparator(evaluation, conjectures[gid]['value']):
                        conjectures[gid] = {'graph6': g.graph6_string(),'value': evaluation, 'expressions': [expr]}

            if not conjectures:
                self.complexity += 1

        print conjectures


    def expressions(self, complexity=None, _cache=None):
        r"""
        Return all possible expressions of the given complexity. If complexity
        is not specified, then use the brain's current complexity level.

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
        if complexity is None:
            complexity = self.complexity

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
