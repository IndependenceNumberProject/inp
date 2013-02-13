import sys
sys.path.append(".") # Needed to pass automated testing.

import collections
import itertools
import operator
import math
from functools import wraps
from inp import INPGraph
from sage.all import *

class GraphBrain(SageObject):
    _complexity_limit = 6
    _graph_db = [INPGraph(graphs.CompleteGraph(3)), INPGraph.KillerGraph(), INPGraph(graphs.CompleteGraph(5))]

    def __init__(self):
        pass

    def _repr_(self):
        pass

    def _latex_(self):
        pass

    def memoize(func):
        func._cache = {}
        @wraps(func)
        def wrap(*args):
            if args not in func._cache:
                func._cache[args] = func(*args)
            return func._cache[args]
        return wrap

    @classmethod
    def conjecture(cls, comparator, invariant):
        """
        Return a list of true statements that are also significant for at least
        one graph in the database.
        """
        rhs = GraphExpression([invariant])

        expressions = []
        for c in range(1, cls._complexity_limit + 1):
            expressions += GraphExpression.all_expressions(c, without=invariant)

        statements = [GraphStatement(x, comparator, rhs) for x in expressions]

        conjectures = {}

        for s in statements:
            if not all(s.evaluate(g) for g in cls._graph_db):
                continue

            for g in cls._graph_db:

                bound = s.lhs.evaluate(g)

                # print s, g, bound

                gid = id(g)

                if gid in conjectures:

                    if s.comparator in [operator.le, operator.lt]:
                        
                        if bound > conjectures[gid]['bound']:
                            conjectures[gid]  = {'bound': bound, 'complexity': s.lhs.complexity(), 'graph': g, 'statements': [s]}
                        elif bound == conjectures[gid]['bound']:

                            if s.lhs.complexity() < conjectures[gid]['complexity']:
                                conjectures[gid]  = {'bound': bound, 'complexity': s.lhs.complexity(), 'graph': g, 'statements': [s]}
                            elif s.lhs.complexity() == conjectures[gid]['complexity']:
                                conjectures[gid]['statements'].append(s)

                    elif s.comparator in [operator.ge, operator.gt]:

                        if bound < conjectures[gid]['bound']:
                            conjectures[gid]  = {'bound': bound, 'complexity': s.lhs.complexity(), 'graph': g, 'statements': [s]}
                        elif bound == conjectures[gid]['bound']:

                            if s.lhs.complexity() < conjectures[gid]['complexity']:
                                conjectures[gid]  = {'bound': bound, 'complexity': s.lhs.complexity(), 'graph': g, 'statements': [s]}
                            elif s.lhs.complexity() == conjectures[gid]['complexity']:
                                conjectures[gid]['statements'].append(s)

                    else:

                        conjectures[gid]['statements'].append(s)

                else:

                    conjectures[gid]  = {
                        'bound': bound,
                        'complexity': s.lhs.complexity(),
                        'graph': g,
                        'statements': [s]
                    }

        for key, data in conjectures.iteritems():
            print data['graph']

            for s in data['statements']:
                print "\t", s

            print

        # return conjectures

class GraphStatement(SageObject):
    _latex_dict = {
        operator.lt: "<",
        operator.le: "\\leq",
        operator.eq: "=",
        operator.ne: "\\not\\eq",
        operator.ge: "\\geq",
        operator.gt: ">"
    }

    _repr_dict = {
        operator.lt: "<",
        operator.le: "<=",
        operator.eq: "=",
        operator.ne: "!=",
        operator.ge: ">=",
        operator.gt: ">"
    }

    def __init__(self, lhs, comparator, rhs):
        r"""
        Constructs a new GraphStatement from an expression stack, a comparator,
        and another expression.
        """
        if isinstance(lhs, GraphExpression):
            self.lhs = lhs
        else:
            self.lhs = GraphExpression([lhs])

        self.comparator = comparator

        if isinstance(rhs, GraphExpression):
            self.rhs = rhs
        else:
            self.rhs = GraphExpression([rhs])

        super(GraphStatement, self).__init__()

    def _repr_(self):
        r"""
        Returns the string representation of the statement.
        """
        return "{0} {1} {2}".format(self.lhs, self._repr_dict[self.comparator], self.rhs)

    def _latex_(self):
        r"""
        Returns the LaTeX representation of the statement.
        """
        return "{0} {1} {2}".format(latex(self.lhs), self._latex_dict[self.comparator], latex(self.rhs))

    def evaluate(self, g):
        """
        Compares the evaluation of the LHS to the evaluation of the RHS. Returns True or False.
        """
        try:
            return self.comparator(self.lhs.evaluate(g), self.rhs.evaluate(g))
        except (ValueError, ZeroDivisionError, OverflowError):
            return False

class GraphExpression(SageObject):
    increment = lambda x: x + 1
    decrement = lambda x: x - 1
    add_constant = lambda x, c: x + c
    sub_constant = lambda x, c: x - c
    reciprocal = lambda x: operator.truediv(1, x)

    _graph_invariants = [INPGraph.alpha, INPGraph.min_degree]
    _unary_operators = [math.sqrt, increment]
    _binary_commutative_operators = [operator.add, operator.mul]
    _binary_noncommutative_operators = [operator.sub, operator.truediv, operator.pow]

    _latex_dict = {
        operator.add: lambda a, b: "{0} + {1}".format(a, b),
        operator.sub: lambda a, b: "{0} - {1}".format(a, b),
        operator.mul: lambda a, b: "{0} * {1}".format(a, b),
        operator.truediv: lambda a, b: "\\frac{{{0}}}{{{1}}}".format(a, b),
        operator.pow: lambda a, b: "{0}^{{{1}}}".format(a, b),
        math.sqrt: lambda a: "\\sqrt{{{0}}}".format(a),
        INPGraph.min_degree: lambda a: "\\delta({0})".format(a),
        INPGraph.alpha: lambda a: "\\alpha({0})".format(a),
        increment: lambda a: "{0} + 1".format(a),
        decrement: lambda a: "{0} - 1".format(a),
        add_constant: lambda a, c: "{0} + {1}".format(a, c),
        sub_constant: lambda a, c: "{0} - {1}".format(a, c),
        reciprocal: lambda a: "\\frac{{1}}{{{0}}}".format(a),
    }

    _repr_dict = {
        operator.add: lambda a, b: "{0} + {1}".format(a, b),
        operator.sub: lambda a, b: "{0} - {1}".format(a, b),
        operator.mul: lambda a, b: "{0} * {1}".format(a, b),
        operator.truediv: lambda a, b: "({0})/({1})".format(a, b),
        operator.pow: lambda a, b: "({0})^({1})".format(a, b),
        math.sqrt: lambda a: "sqrt({0})".format(a),
        INPGraph.min_degree: lambda a: "delta({0})".format(a),
        INPGraph.alpha: lambda a: "alpha({0})".format(a),
        increment: lambda a: "{0} + 1".format(a),
        decrement: lambda a: "{0} - 1".format(a),
        add_constant: lambda a, c: "{0} + {1}".format(a, c),
        sub_constant: lambda a, c: "{0} - {1}".format(a, c),
        reciprocal: lambda a: "1/({0})".format(a)
    }

    def __init__(self, stack):
        r"""
        Constructs a new GraphExpression from the given stack of functions.
        """
        self._stack = stack
        super(GraphExpression, self).__init__()

    def _repr_(self):
        """
        Returns the string representation of the expression.

        EXAMPLES:

            ::
                sage: exp = GraphExpression([INPGraph.alpha])
                sage: exp
                alpha(G)
                sage: exp = GraphExpression([INPGraph.min_degree, math.sqrt])
                sage: exp
                sqrt(delta(G))
        """
        stack = list(self._stack)

        for i, op in enumerate(self._stack):
            if op in self._graph_invariants:
                # stack.append("{0}(G)".format(self._repr_dict[op]))
                stack.append(self._repr_dict[op]("G"))
            
            elif op in self._unary_operators:
                # stack.append("{0}({1})".format(self._repr_dict[op], stack.pop()))
                stack.append(self._repr_dict[op](stack.pop()))

            elif op in self._binary_commutative_operators or op in self._binary_noncommutative_operators:
                # We don't need parens if it's the final expression.
                if i == len(self._stack) - 1:
                    # stack.append("{0} {1} {2}".format(stack.pop(), self._repr_dict[op], stack.pop()))
                    stack.append(self._repr_dict[op](stack.pop(), stack.pop()))
                else:
                    # stack.append("({0} {1} {2})".format(stack.pop(), self._repr_dict[op], stack.pop()))
                    stack.append("({0})".format(self._repr_dict[op](stack.pop(), stack.pop())))

        return "{0}".format(stack.pop())

    def _latex_(self):
        """
        Returns the LaTeX representation of the expression.

        EXAMPLES:

            ::
                sage: exp = GraphExpression([INPGraph.alpha])
                sage: latex(exp)
                \alpha(G)
                sage: exp = GraphExpression([INPGraph.min_degree, math.sqrt])
                sage: latex(exp)
                \sqrt{\delta(G)}
        """
        stack = list(self._stack)

        for i, op in enumerate(self._stack):
            if op in self._graph_invariants:
                #stack.append("{0}(G)".format(self._latex_dict[op]))
                stack.append(self._latex_dict[op]("G"))
            
            elif op in self._unary_operators:
                # stack.append("{0}{{{1}}}".format(self._latex_dict[op], stack.pop()))
                stack.append(self._latex_dict[op](stack.pop()))

            elif op in self._binary_commutative_operators or op in self._binary_noncommutative_operators:
                # We don't need parens if it's the final expression.
                if i == len(self._stack) - 1:
                    # stack.append("{0} {1} {2}".format(stack.pop(), self._latex_dict[op], stack.pop()))
                    stack.append(self._latex_dict[op](stack.pop(), stack.pop()))
                else:
                    # stack.append("\\left({0} {1} {2}\\right)".format(stack.pop(), self._latex_dict[op], stack.pop()))
                    stack.append("\\left({0}\\right)".format(self._latex_dict[op](stack.pop(), stack.pop())))

        return stack.pop()

    def append(self, x):
        """
        Append a command to the right end of the expression stack.
        """
        return GraphExpression(self._stack + [x])

    def extend(self, li):
        """
        Extend the expression stack by the given list.
        """
        return GraphExpression(self._stack + li)

    def complexity(self):
        """
        Return the complexity of the expression, i.e. the length of the stack.
        """
        return len(self._stack)

    def evaluate(self, g):
        r"""
        Evaluate the expression for the given graph.

        EXAMPLES:

            ::
                sage: exp = GraphExpression([INPGraph.alpha])
                sage: g = INPGraph(graphs.PetersenGraph())
                sage: exp.evaluate(g)
                4
                sage: exp = GraphExpression([INPGraph.min_degree, math.sqrt])
                sage: exp.evaluate(g)
                1.7320508075688772
        """
        stack = []

        for op in self._stack:
            if op in self._graph_invariants:
                stack.append(op(g))
            elif op in self._unary_operators:
                stack.append(op(stack.pop()))
            elif op in self._binary_commutative_operators or op in self._binary_noncommutative_operators:
                stack.append(op(stack.pop(), stack.pop()))

        return stack.pop()

    @classmethod
    def all_expressions(cls, complexity, without=None):
        """
        Generate all expressions of a given complexity.
        """
        if complexity < 1:
            return GraphExpression([])
        elif complexity == 1:
            return [GraphExpression([f]) for f in cls._graph_invariants if not f == without]
        else:
            new_expressions = []

            # Apply a unary operator to strings of complexity-1, e.g.
            # if we want strings of complexity 5, we can square root all the
            # strings of complexity 4.
            for s in cls.all_expressions(complexity - 1, without):
                for op in cls._unary_operators:
                    new_expressions += [s.append(op)]

            # Apply binary noncommutative operators, if we want strings of
            # complexity 5, we need to apply to strings of the following complexity
            # combinations: 1,3  2,2  3,1
            for i in range(1, complexity - 1):
                strings_a = cls.all_expressions(i, without)
                strings_b = cls.all_expressions(complexity - 1 - i, without)
                for a in strings_a:
                    for b in strings_b:
                        for op in cls._binary_noncommutative_operators:

                            # Skip subtracting or dividing something from itself
                            if a._stack == b._stack and op in [operator.sub, operator.truediv]:
                                continue

                            #new_expressions += GraphExpression([a + b + [op]])
                            new_expressions += [a.extend(b._stack).append(op)]

            # Apply binary commutative operators, since they are commutative we
            # only need to check each combination of lower complexities once, e.g.
            # for strings of complexity 6, we need to work on: 1,4  2,3
            for k in range(1, ceil(float(complexity)/2)):
                strings_a = cls.all_expressions(k, without)
                if k == complexity - 1 - k:
                    # If lhs and rhs have the same complexity, since this is commutative,
                    # we can skip the duplicates.
                    for a, b in itertools.combinations_with_replacement(strings_a, 2):
                        for op in cls._binary_commutative_operators:
                            new_expressions += [a.extend(b._stack).append(op)]
                else:
                    # If lhs and rhs have different complexity, we have to check
                    # all possible pairings.
                    strings_b = cls.all_expressions(complexity - 1 - k, without)
                    for a in strings_a:
                        for b in strings_b:
                            for op in cls._binary_commutative_operators:
                                # new_expressions += [a + b + [op]]
                                new_expressions += [a.extend(b._stack).append(op)]

            return new_expressions
