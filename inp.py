from sys.stdout import write

def difficult_graph_search():
    n = 1
    while True:
        print 'Graph{0} on {1} vert{2}'.format(['', 's'][n != 1], n, ['ex', 'ices'][n != 1])
        gen = graphs.nauty_geng(str(n))
        while True:
            try:
                g = gen.next()
                sys.stdout.write('.')
                if is_difficult(g):
                    print
                    return g
            except StopIteration:
                n += 1
                print
                break

def is_difficult(g):
    if has_alpha_property(g):
        return false

    lbound = lower_bound(g)
    ubound = upper_bound(g)

    if lbound == ubound:
        return false

    return true

def has_alpha_property(g):
    return true

def lower_bound(g):
    return 1

def upper_bound(g):
    return g.num_verts()