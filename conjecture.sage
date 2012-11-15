from operator import add, mul, sub
from math import floor, sqrt
from inp import INPGraph

#attach inp.py

invariants = [INPGraph.alpha, INPGraph.min_degree]

unary_ops = [sqrt]
binary_commutative_ops = [add, mul]
binary_noncommutative_ops = [sub]

def get_strings(complexity):
    if complexity < 1:
        return []
    elif complexity == 1:
        return [[f] for f in invariants]
    else:
        new_strings = []

        for s in get_strings(complexity - 1):
            for op in unary_ops:
                new_strings += [s + [op]]

        for i in range(1, complexity - 1):
            strings_a = get_strings(i)
            strings_b = get_strings(complexity - 1 - i)
            for a in strings_a:
                for b in strings_b:
                    for op in binary_noncommutative_ops:

                        if op == sub and a == b:
                            continue

                        new_strings += [a + b + [op]]

        for k in range(1, ceil(float(complexity)/2)):
            strings_a = get_strings(i)
            if k == complexity - 1 - k:
                for i, a in enumerate(strings_a):
                    for b in strings_a[i:]:
                        for op in binary_commutative_ops:
                            new_strings += [a + b + [op]]
            else:
                strings_b = get_strings(complexity - 1 - k)
                for a in strings_a:
                    for b in strings_b:
                        for op in binary_commutative_ops:
                            new_strings += [a + b + [op]]


        return new_strings

def eval_string(s, g):
    stack = []
    for op in s:
        if op in invariants:
            stack.append(op(g))
        elif op in unary_ops:
            stack[-1] = [op(stack[-1])]
        elif op in binary_commutative_ops or op in binary_noncommutative_ops:
            stack[-2:] = [op(*stack[-2:])]
    return stack.pop()

latex_dict = {
    add: "+",
    sub: "-",
    mul: "*",
    sqrt: "\\sqrt",
    INPGraph.min_degree: "\\delta",
    INPGraph.alpha: "\\alpha"
}

def latex_string(s):
    stack = []
    for (i, op) in enumerate(s):
        if op in invariants:
            stack.append(latex_dict[op] + "(G)")
        elif op in unary_ops:
            stack.append(latex_dict[op] + "{" + stack.pop() + "}")
        elif op in binary_commutative_ops or op in binary_noncommutative_ops:
            # We don't need parens if it's the final expression.
            if i == len(s) - 1:
                stack.append(stack.pop() + ' ' + latex_dict[op] + ' ' + stack.pop())
            else:
                stack.append("\\left(" + stack.pop() + ' ' + latex_dict[op] + ' ' + stack.pop() + "\\right)")
    return stack.pop()

def itemize_all(n):
    print "\\begin{itemize}\n" + \
        "\n".join(["\\item $"+latex_string(s)+"$" for s in get_strings(n)]) + \
        "\n\\end{itemize}"