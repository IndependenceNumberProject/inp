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

    # _default_graph_invariants =[Graph.diameter, Graph.radius]
    _default_unary_operators = [sqrt]
    _default_binary_commutative_operators = [operator.add, operator.mul]
    _default_binary_noncommutative_operators = [operator.sub, operator.truediv]

    _complexity_limit = 10

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

    def _repr_(self):
        return "Name: {0}\nComparator: {1}\nGraphs: {2}\nTarget: {3}\nGraph invariants: {4}\nUnary operators: {5}\nBinary commutative operators: {6}\nBinary noncommutative operators:{7}".format(
            self.name, self.comparator, self.graphs, self.target, self.graph_invariants, self.unary_operators, self.binary_commutative_operators, self.binary_noncommutative_operators)

    def conjecture(self, verbose=True, debug=False):
        r"""
        Return a list of true statements that are also significant for at least
        one graph in the brain, that is, the statement gives the tightest bound.
        """
        if not self.graphs:
            raise ValueError("There must be at least one graph in the brain.")

        if debug: verbose = False
        complexity = 1

        targets = {}
        bingos = {}
        significance = {}

        while not bingos or not all(bingos.values()) and complexity <= self._complexity_limit:

            if debug: print "========== COMPLEXITY", complexity, "=========="

            expressions = self.expressions(complexity)
            expression_count = len(expressions)
            counter = 0

            if debug: print expressions

            for expr in expressions:

                if debug: print expr

                exprid = id(expr)
                truth = []
                possible_significance = {}
                possible_bingos = {}

                try:
                    for g in self.graphs:

                        gid = id(g)

                        if debug: print "\t->", g.graph6_string(), "=",expr.evaluate(g)

                        if gid not in targets:
                            targets[gid] = self.target(g)

                        if gid not in bingos:
                            bingos[gid] = False

                        evaluation = expr.evaluate(g)
                        target = targets[gid]
                        true_for_this_graph = self.comparator(evaluation, target)
                        truth.append(true_for_this_graph)

                        if true_for_this_graph:
                            if gid not in significance or \
                                (self.comparator in [operator.gt, operator.ge] and evaluation < significance[gid]['value']) or \
                                (self.comparator in [operator.lt, operator.le] and evaluation > significance[gid]['value']):

                                if debug: print "\t\tPossible significance"
                                if exprid not in possible_significance:
                                    possible_significance[exprid] = {}
                                possible_significance[exprid][gid] = {'exprid': exprid, 'expression': expr, 'value': evaluation}
                            elif self.comparator not in [operator.gt, operator.ge, operator.lt, operator.le]:
                                raise ValueError("Significance is not defined for this comparator.")

                            if evaluation == target:
                                if debug: print "\t\tPossible bingo"
                                if exprid not in possible_bingos:
                                    possible_bingos[exprid] = {}
                                possible_bingos[exprid][gid] = True
                except Exception as e:
                    if debug: print e
                    if debug: print "Exception, bailing out of graph loop."

                if debug: print "\tTrue for all graphs:", all(truth)

                if all(truth) and exprid in possible_significance:
                    for gid in possible_significance[exprid]:
                        significance[gid] = possible_significance[exprid][gid]

                if debug: print "\tPossible bingos:", possible_bingos
                        
                if all(truth) and exprid in possible_bingos:
                    for gid in possible_bingos[exprid]:
                        if possible_bingos[exprid][gid]:
                            bingos[gid] = True
                            if debug: print "\tBingo added"

                if debug: print "\tSignificant:", significance
                if debug: print "\tBingos:", bingos
                if debug: print

                counter += 1
                num_bingos = sum(1 for gid in bingos if bingos[gid])
                if verbose:
                    sys.stdout.write("\rSearching complexity {0}: {1}/{2} ({3:.2f}%) (Bingos: {4}/{5})".format(complexity, counter, expression_count, (float(counter)/expression_count)*100, num_bingos, len(bingos)))
                    sys.stdout.flush()

                if all(bingos.values()): break

            complexity += 1
            if verbose: print

        conjectures = {}
        for gid in significance:
            conjectures[significance[gid]['exprid']] = significance[gid]['expression']

        return conjectures.values()


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
            [diameter(G)^(1/4), radius(G)^(1/4), 2*diameter(G), radius(G) - diameter(G), -radius(G) + diameter(G), radius(G) + diameter(G), 2*radius(G)]
        """
        brain_tuple = tuple([tuple(self.graph_invariants), tuple(self.unary_operators),
            tuple(self.binary_commutative_operators), tuple(self.binary_noncommutative_operators), self.target])

        if brain_tuple not in self.expressions._cache:
            self.expressions._cache[brain_tuple] = {}

        if complexity not in self.expressions._cache[brain_tuple]:
            if complexity < 1:
                return []
            elif complexity == 1:
                self.expressions._cache[brain_tuple][1] = [GraphExpression(self, [inv]) for inv in self.graph_invariants if inv != self.target]
            else:
                self.expressions._cache[brain_tuple][complexity] = []

                # Unary operators
                for expr in self.expressions(complexity - 1, _cache):
                    for op in self.unary_operators:
                        self.expressions._cache[brain_tuple][complexity].append(expr.operate(op))

                # Binary operators
                for k in range(1, complexity - 1):
                    for i, a in enumerate(self.expressions(k, _cache)):
                        for j, b in enumerate(self.expressions(complexity - 1 - k, _cache)):
                            # Noncommutative
                            for op in self.binary_noncommutative_operators:
                                new_expr = a.operate(op, b)
                                if not new_expr.expression().is_numeric():
                                    self.expressions._cache[brain_tuple][complexity].append(a.operate(op, b))

                            # Commutative
                            if k <= complexity - 1 - k and j <= i:
                                for op in self.binary_commutative_operators:
                                    new_expr = a.operate(op, b)
                                    if not new_expr.expression().is_numeric():
                                        self.expressions._cache[brain_tuple][complexity].append(a.operate(op, b))

        return self.expressions._cache[brain_tuple][complexity]
    expressions._cache = {}
        
class GraphExpression(SageObject):

    def __init__(self, brain, rpn_stack):
        """Constructs a new GraphExpression from the given stack of functions."""
        self.brain = brain
        self.rpn_stack = rpn_stack
        super(GraphExpression, self).__init__()

    def __eq__(self, other):
        return self.rpn_stack == other.rpn_stack

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
            except (ValueError, ZeroDivisionError, sage.rings.infinity.SignError) as e:
                raise ValueError("Can't evaluate", self.rpn_stack, ":", e)
                # print "Can't evaluate", self.rpn_stack, ":", e
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
                try:
                    if op in self.brain.graph_invariants:
                        stack.append(function(op.__name__, g, evalf_func=op))
                    elif op in self.brain.unary_operators:
                        stack.append(op(stack.pop()))
                    elif op in self.brain.binary_commutative_operators + self.brain.binary_noncommutative_operators:
                        stack.append(op(stack.pop(), stack.pop()))
                    else:
                        raise ValueError("Expression stack contains something the brain doesn't understand.")
                except (ValueError, ZeroDivisionError, sage.rings.infinity.SignError) as e:
                    raise ValueError("Can't display", self.rpn_stack, ":", e)
                    # print "Can't display", self.rpn_stack, ":", e
                    return None

            return stack.pop()
        else:
            return None

