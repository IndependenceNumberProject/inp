import sys
import inspect

import alphaproperties
import lowerbounds
import upperbounds

def difficult_graph_search():
    n = 1
    while True:
        print 'Graph{0} on {1} vert{2}'.format(['', 's'][n != 1], n, ['ex', 'ices'][n != 1])
        gen = graphs.nauty_geng(str(n))
        while True:
            try:
                g = gen.next()
                sys.stdout.write('.')
                sys.stdout.flush()
                if is_difficult(g):
                    print
                    return g
            except StopIteration:
                n += 1
                print
                break

def is_difficult(g):
    if has_alpha_property(g):
        return False

    lbound = lower_bound(g)
    ubound = upper_bound(g)

    if lbound == ubound:
        return False

    return True

def has_alpha_property(g):
    for name, obj in inspect.getmembers(alphaproperties, inspect.isclass):
        if obj.__module__ == 'alphaproperties':
            if obj.has_property(g):
                return True
    return False

def lower_bound(g):
    lbound = 1
    for name, obj in inspect.getmembers(lowerbounds, inspect.isclass):
        if obj.__module__ == 'lowerbounds':
            new_bound = obj.bound(g)
            if new_bound > lbound:
                lbound = new_bound
    return lbound

def upper_bound(g):
    ubound = g.num_verts()
    for name, obj in inspect.getmembers(upperbounds, inspect.isclass):
        if obj.__module__ == 'upperbounds':
            new_bound = obj.bound(g)
            if new_bound < ubound:
                ubound = new_bound
    return ubound