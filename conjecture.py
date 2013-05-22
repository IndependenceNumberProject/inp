import sys
sys.path.append(".") # Needed to pass Sage's automated testing

from sage.all import *
from inp import INPGraph
import itertools
import operator

class GraphBrain(SageObject):
    # Don't add Graph.wiener_index to the graph invariants, it causes a bug when
    # creating a symbolic function in GraphExpression.expression() for reasons unknown.
    # TODO: Fix whatever is causing wiener_index to break the expression code.
    _default_graph_invariants = [Graph.average_distance, Graph.diameter, Graph.radius, Graph.girth,
                                 INPGraph.matching_number, Graph.order, Graph.size, Graph.szeged_index,
                                 INPGraph.residue, INPGraph.fractional_alpha,
                                 INPGraph.annihilation_number, INPGraph.lovasz_theta, INPGraph.cvetkovic,
                                 INPGraph.max_degree, INPGraph.min_degree, Graph.average_degree]
    _default_unary_operators = [sqrt]
    _default_binary_commutative_operators = [operator.add, operator.mul]
    _default_binary_noncommutative_operators = [operator.sub, operator.truediv]

    _complexity_limit = 3

    _save_path = os.path.expanduser("~/Dropbox/INP")

    def __init__(self, name=None, comparator=operator.le, graphs=[],
                 target=INPGraph.independence_number,
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

        self.graph_invariants = graph_invariants
        self.unary_operators = unary_operators
        self.binary_commutative_operators = binary_commutative_operators
        self.binary_noncommutative_operators = binary_noncommutative_operators
        self.conjectures = []
        self.conjecture_cache = {}

    def conjecture(self, verbose=True):
        r"""
        Return a list of true statements that are also significant for at least
        one graph in the brain, that is, the statement gives the tightest bound.
        """
        if not self.graphs:
            raise ValueError("There must be at least one graph in the brain.")

        bingos = {id(g): False for g in self.graphs}
        complexity = 1
        while not all(bingos.values()) and complexity <= self._complexity_limit:
            if verbose: print "*********************", complexity, "*********************"
            for expr in self.expressions(complexity):
                if verbose: print expr
                if verbose: print "\tComplexity:", complexity
                if verbose: print "\tAlpha:", [self.target(g) for g in self.graphs]
                if verbose: print "\tEvals:", [N(expr.evaluate(g)) for g in self.graphs]
                if verbose: print "\tNew bingos:", expr.get_bingos()
                if verbose: print "\tConsistent:", expr.is_consistent()
                if verbose: print "\tSignificant:", expr.is_significant()
                if expr.is_consistent() and expr.is_significant():
                    if verbose: print "\t*** Adding conjecture to brain."
                    brain.add_conjecture(expr)
                bingos.update(expr.get_bingos())
                #if verbose: print "\tConjectures:", self.conjectures
                if verbose: print "\tAll bingos:", bingos
                if verbose: print "\n"
            complexity += 1
        return self.conjectures

    def expressions(self, complexity, _cache=None):
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

    def add_conjecture(self, expr):
        self.conjectures.append(expr)
        for g in self.graphs:
            if id(g) not in self.conjecture_cache:
                self.conjecture_cache[id(g)] = {}

            self.conjecture_cache[id(g)][id(expr)] = expr.evaluate(g)
        self._remove_insignificant_conjectures()

    def _remove_insignificant_conjectures(self):
        for expr in self.conjectures:
            if not expr.is_significant():
                self.conjectures.remove(expr)
                for g in self.graphs:
                    del self.conjecture_cache[id(g)][id(expr)]
        
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
            try:
                if op in self.brain.graph_invariants:
                    stack.append(op(g))
                elif op in self.brain.unary_operators:
                    stack.append(op(stack.pop()))
                elif op in self.brain.binary_commutative_operators + self.brain.binary_noncommutative_operators:
                    stack.append(op(stack.pop(), stack.pop()))
            except (ValueError, sage.rings.infinity.SignError) as e:
                print "Can't evaluate", self, ":", e
                return None
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

        if self.rpn_stack:
            g = var(graph_variable) 
            stack = []

            for op in self.rpn_stack:
                if op in self.brain.graph_invariants:
                    stack.append(function(op.__name__, g, evalf_func=op))
                elif op in self.brain.unary_operators:
                    stack.append(op(stack.pop()))
                elif op in self.brain.binary_commutative_operators + self.brain.binary_noncommutative_operators:
                    stack.append(op(stack.pop(), stack.pop()))
                else:
                    raise ValueError("Expression stack contains something the brain doesn't understand.")

            return stack.pop()
        else:
            return None

    def is_true(self, g):
        r"""
        Return True when the expression is compared (using the brain's comparator)
        to the target invariant for the given graph.
        """
        return self.brain.comparator(self.evaluate(g), self.brain.target(g))

    def is_consistent(self):
        r"""
        Return True if the expression is true for each graph stored in the brain.
        """
        # all(self.comparator(evaluations[id(g)], targets[id(g)]) for g in self.graphs):
        #evaluations = (self.evaluate(g) for g in self.brain.graphs)
        #print evaluations
        #return False
        return all(self.is_true(g) for g in self.brain.graphs)

    def is_significant(self):
        for g in self.brain.graphs:
            if not self.brain.conjectures: return True

            if self.brain.comparator in [operator.lt, operator.le]:
                if self.evaluate(g) >= max(self.brain.conjecture_cache[id(g)].values()):
                    return True
            elif self.brain.comparator in [operator.gt, operator.ge]:
                if self.evaluate(g) <= min(self.brain.conjecture_cache[id(g)].values()):
                    return True
            else:
                raise ValueError("Significance is not defined for this comparator.")
        return False

    def get_bingos(self):
        r"""
        Return a dictionary containing the id() of graphs (with the value True)
        for which the expression is equal to the target invariant.
        """
        return {id(g): True for g in self.brain.graphs if self.evaluate(g) == self.brain.target(g)}
