r"""
Search for difficult graphs in the Independence Number Project.

AUTHORS:

- Patrick Gaskill (2012-08-21): initial version
"""

#*****************************************************************************
#       Copyright (C) 2012 Patrick Gaskill <gaskillpw@vcu.edu>
#       Copyright (C) 2012 Craig Larson <clarson@vcu.edu>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#*****************************************************************************

import datetime
import inspect
from string import Template
import subprocess
import sys
import time

def difficult_graph_search(verbose=True):
    r"""
    This function returns the smallest graph considered difficult by INP theory.

    INPUT:

    - ``verbose`` - boolean -- Print progress to the console and save graph
        information as a dossier PDF and a PNG image.

    OUTPUT:

    sage.graphs.Graph -- the first difficult graph encountered in the order
    given by `nauty_geng`.

    EXAMPLES:

    ::

        sage: G = difficult_graph_search(verbose=False) # long time
        sage: isinstance(G, Graph) # long time
        True

    ::

        sage: G = difficult_graph_search(verbose=False) # long time
        sage: is_difficult(G) # long time
        True

    NOTES:

    The return value of this function may change depending on the functions
    included in the AlphaProperties, LowerBounds, and UpperBounds classes.
    """
    n = 1
    count = 0

    while True:

        if verbose:
            print 'Graph{0} on {1} vert{2}'.format(['', 's'][n != 1],
                                                    n, ['ex', 'ices'][n != 1])
        gen = graphs.nauty_geng("{0} -c".format(n))
        while True:
            try:
                g = gen.next()
                count += 1

                if verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                if is_difficult(g):

                    if verbose:
                        print "\n\nFound a difficult graph: {0}".format(g.graph6_string())

                        print "{0} graphs searched.".format(count)

                        filename = "difficult_graph_{0}_{1}".format(n,
                            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

                        filepath = os.path.expanduser("~/Dropbox/INP/{0}".format(filename))

                        try:
                            if not os.path.exists(filepath):
                                os.makedirs(filepath)
                        except IOError:
                            "Can't make directory {0}".format(filepath)

                        try:
                            p = g.plot()
                            p.save("{0}/{1}.png".format(filepath, filename))
                            print "Plot saved."
                        except IOError:
                            print "Couldn't save {0}.png".format(filename)

                        try:
                            _export_latex_pdf(g, filepath, filename)
                            print "Dossier saved."
                        except IOError:
                            print "Couldn't save {0}.pdf".format(filename)

                    return g

            except StopIteration:
                if verbose:
                    print

                n += 1
                break

def _export_latex_pdf(g, filepath, filename):
    # Generate the latex for the information box
    info_table = """
    \\rowcolor{{LightGray}} $n$ & {0} \\\\
    \\rowcolor{{LightGray}} $e$ & {1} \\\\
    \\rowcolor{{LightGray}} $\\alpha$ & {2} \\\\
    """.format(g.num_verts(), g.num_edges(), len(g.independent_set()))

    # Generate the latex for the lower bounds table
    lowerbounds_table = ''
    for name, func in inspect.getmembers(LowerBounds, inspect.isfunction):
        value = func(g)
        if value in ZZ:
            lowerbounds_table += \
                "{0} & {1} \\\\\n".format(name, value).replace('_', '\_')
        else:
            lowerbounds_table += \
                "{0} & {1:.4f} \\\\\n".format(name, value).replace('_', '\_')

    # Generate the latex for the upper bounds table
    upperbounds_table = ''
    for name, func in inspect.getmembers(UpperBounds, inspect.isfunction):
        value = func(g)
        if value in ZZ:
            upperbounds_table += \
                "{0} & {1} \\\\\n".format(name, value).replace('_', '\_')
        else:
            upperbounds_table += \
                "{0} & {1:.4f} \\\\\n".format(name, value).replace('_', '\_')

    # Generate the latex for the alpha properties table
    alphaproperties_table = ''
    for name, func in inspect.getmembers(AlphaProperties, inspect.isfunction):
        alphaproperties_table += \
            "{0} \\\\\n".format(name).replace('_', '\_')

    # Insert all the generated latex into the template file
    template_file = open('dossier_template.tex', 'r')
    template = template_file.read()
    s = Template(template)

    output = s.substitute(graph=latex(g), 
                          name=g.graph6_string().replace('_', '\_'),
                          info=info_table,
                          lowerbounds=lowerbounds_table, 
                          upperbounds=upperbounds_table,
                          alphaproperties=alphaproperties_table)
    latex_filename = "{0}/{1}.tex".format(filepath, filename)

    # Write the latex to a file then run pdflatex on it
    try:
        latex_file = open(latex_filename, 'w')
        latex_file.write(output)
        latex_file.close()
        with open(os.devnull, 'wb') as devnull:
            subprocess.call(['/usr/texbin/pdflatex', '-output-directory',
                filepath, latex_filename],
                stdout=devnull, stderr=subprocess.STDOUT)
    except:
        pass

def is_difficult(g):
    r"""
    This function determines if a given Graph `g` is difficult as described by
    INP theory.

    EXAMPLES:

    ::

        sage: G = Graph(1)
        sage: is_difficult(G)
        False

    NOTES:

    The return value of this function may change depending on the functions
    included in the AlphaProperties, LowerBounds, and UpperBounds classes.
    """
    if has_alpha_property(g):
        return False

    lbound = ceil(lower_bound(g))
    ubound = floor(upper_bound(g))

    if lbound == ubound:
        return False

    return True

def has_alpha_property(g):
    r"""
    This function determines if a given Graph `g` satisifes any of the known
    alpha-properties.

    EXAMPLES:

    ::

        sage: G = difficult_graph_search(verbose=False) # long time
        sage: has_alpha_property(G) # long time
        False

    NOTES:

    The return value of this function may change depending on the functions
    included in the AlphaProperties class.
    """

    # Loop through all the functions in the AlphaProperties class
    for name, func in inspect.getmembers(AlphaProperties, inspect.isfunction):
        if func(g):
            return True
    return False

def lower_bound(g):
    r"""
    This function computes a lower bound for the independence number of the
    given graph `g`.

    EXAMPLES:

    ::

        sage: lower_bound(Graph(1))
        1

    ::

        sage: G = graphs.CompleteGraph(3)
        sage: lbound = lower_bound(G)
        sage: lbound >= 1 and lbound <= G.num_verts()
        True

    NOTES:

    The return value of this function may change depending on the functions
    included in the LowerBounds class.
    """

    # The default bound is 1
    lbound = 1

    # Loop through all the functions in LowerBounds class
    for name, func in inspect.getmembers(LowerBounds, inspect.isfunction):
        new_bound = func(g)
        if new_bound > lbound:
            lbound = new_bound
    return lbound

def upper_bound(g):
    r"""
    This function computes an upper bound for the independence number of the
    given graph `g`.

    EXAMPLES:

    ::

        sage: upper_bound(Graph(1))
        1

    ::

        sage: G = graphs.CompleteGraph(3)
        sage: ubound = upper_bound(G)
        sage: ubound >= 1 and ubound <= G.num_verts()
        True

    NOTES:

    The return value of this function may change depending on the functions
    included in the UpperBounds class.
    """

    # The default upper bound is the number of vertices
    ubound = g.num_verts()

    # Loop through all the functions in UpperBounds class
    for name, func in inspect.getmembers(UpperBounds, inspect.isfunction):
        new_bound = func(g)
        if new_bound < ubound:
            ubound = new_bound
    return ubound

def matching_number(g):
    r"""
    Return the matching number of the graph.

    EXAMPLES:

    ::

        sage: matching_number(graphs.PathGraph(3))
        1

    ::

        sage: matching_number(graphs.PathGraph(4))
        2

    WARNINGS:

    The Sage function matching() function is currently bugged, so this function
    will need to change in Sage v5.3.
    """

    # Sage 5.2 currently returns 2*mu when ignoring edge labels!
    return Integer(g.matching(value_only=True, use_edge_labels=False)/2)

def bidouble(g):
    r"""
    Return a bipartite double cover of g, also known as the bidouble of g.

    EXAMPLES:

    ::

        sage: G = bidouble(graphs.CompleteGraph(3))
        sage: G.is_isomorphic(graphs.CycleGraph(6))
        True

    """

    # We can't use the quick way because we want to preserve vertex numbering
    # return BipartiteGraph(data=g.adjacency_matrix())

    offset = max(g.vertices()) + 1

    b = Graph()
    b.add_vertices(g.vertices() + [v+offset for v in g.vertices()])

    for u, v in g.edge_iterator(labels=False):
        b.add_edges([[u, v+offset], [u+offset, v]])

    return b

def union_MCIS(g):
    r"""
    Return a union of maximum critical independent sets (MCIS) in g.

    EXAMPLES:

    ::

        sage: union_MCIS(Graph('Cx'))
        [0, 1, 3]

    ::

        sage: union_MCIS(graphs.CycleGraph(4))
        [0, 1, 2, 3]

    """

    offset = max(g.vertices()) + 1
    b = bidouble(g)
    # Since b is bipartite, we may use alpha = n - mu
    alpha = b.num_verts() - matching_number(b)

    result = []

    for v in g.vertices():
        to_delete = [v, v+offset] + b.neighbors(v) + b.neighbors(v+offset)
        test_graph = b.copy()
        test_graph.delete_vertices(to_delete)
        alpha_test = test_graph.num_verts() - matching_number(test_graph) + 2
        if alpha_test == alpha:
            result.append(v)

    return result

class AlphaProperties(object):
    @staticmethod
    def has_max_degree_order_minus_one(g):
        r"""
        Determine if the graph has a vertex with degree `n(G)-1`.

        EXAMPLES:

        ::

            sage: AlphaProperties.has_max_degree_order_minus_one(Graph(2))
            False

        ::

            sage: G = graphs.CompleteGraph(3)
            sage: AlphaProperties.has_max_degree_order_minus_one(G)
            True

        """
        return max(g.degree()) == g.num_verts() - 1

    @staticmethod
    def is_claw_free(g):
        r"""
        Determine if the graph contains a claw, i.e., an induced `K_{1,3}`
        subgraph.

        EXAMPLES:

        ::

            sage: AlphaProperties.is_claw_free(Graph(2))
            True

        ::

            sage: AlphaProperties.is_claw_free(graphs.StarGraph(4))
            False

        """
        subsets = combinations_iterator(g.vertices(), 4)
        for subset in subsets:
            if g.subgraph(subset).degree_sequence() == [3,1,1,1]:
                return False
        return True

    @staticmethod
    def has_pendant_vertex(g):
        r"""
        Determine if the graph contains a pendant vertex.

        EXAMPLES:

        ::

            sage: AlphaProperties.has_pendant_vertex(Graph(2))
            False

        ::

            sage: AlphaProperties.has_pendant_vertex(graphs.StarGraph(4))
            True

        """
        return 1 in g.degree_sequence()

    @staticmethod
    def has_complete_closed_neighborhood(g):
        r"""
        Determine if the graph contains a complete closed neighborhood.

        EXAMPLES:

        ::

            sage: G = graphs.CycleGraph(4)
            sage: AlphaProperties.has_complete_closed_neighborhood(G)
            False

        ::

            sage: G = graphs.CompleteGraph(4)
            sage: AlphaProperties.has_complete_closed_neighborhood(G)
            True

        """
        for v in g.vertices():
            if g.subgraph(g.neighbors(v)).is_clique():
                return True

        return False

    @staticmethod
    def is_bipartite(g):
        r"""
        Determine if the graph is bipartite.

        EXAMPLES:

        ::

            sage: AlphaProperties.is_bipartite(graphs.CompleteGraph(4))
            False

        ::

            sage: AlphaProperties.is_bipartite(graphs.PathGraph(3))
            True

        NOTES:

        This property is strictly weaker than is_KE().
        """
        return g.is_bipartite()

    @staticmethod
    def is_KE(g):
        r"""
        Determine if the graph is Konig-Egervary.

        EXAMPLES:

        ::

            sage: AlphaProperties.is_KE(graphs.PathGraph(3))
            True

        ::

            sage: AlphaProperties.is_KE(graphs.CycleGraph(3))
            False

        Not true that `\alpha_f + \mu = n` implies KE ::

            sage: AlphaProperties.is_KE(Graph('GCpvdw'))
            False

        The graph `H_1` from Levit-Mandrescu 2011 is KE ::

            sage: AlphaProperties.is_KE(Graph('Cx'))
            True

        The graph `H_2` from Levit-Mandrescu 2011 is also KE ::

            sage: AlphaProperties.is_KE(Graph('FhcGO'))
            True

        But `H_3` from Levit-Mandrescu 2011 is not KE ::

            sage: AlphaProperties.is_KE(Graph('DxC'))
            False

        """
        
        c = union_MCIS(g)

        nc = []
        for v in c:
            nc.extend(g.neighbors(v))

        return list(set(c + nc)) == g.vertices()

    @staticmethod
    def is_almost_KE(g):
        r"""
        Determine if a graph is almost KE, that is, if for any vertex v,
        G-v is KE.

        EXAMPLES:

        ::

            sage: AlphaProperties.is_almost_KE(graphs.CompleteGraph(3))
            True

        ::

            sage: AlphaProperties.is_almost_KE(graphs.CompleteGraph(4))
            False

        """
        subsets = combinations_iterator(g.vertices(), g.num_verts()-1)
        for subset in subsets:
            if AlphaProperties.is_KE(g.subgraph(subset)):
                return True

        return False

    @staticmethod
    def has_nonempty_KE_part(g):
        r"""
        Determine if the graph contains a nonempty KE part. A graph
        contains a nonempty KE part if 

        """
        if union_MCIS(g):
            return True
        else:
            return False

class LowerBounds(object):
    @staticmethod
    def matching_bound(g):
        r"""
        Compute the matching number lower bound.

        EXAMPLES:

        ::

            sage: LowerBounds.matching_bound(graphs.CompleteGraph(3))
            1

        """
        return g.num_verts() - 2 * matching_number(g)

class UpperBounds(object):
    @staticmethod
    def matching_bound(g):
        r"""
        Compute the matching number upper bound.

        EXAMPLES:

        ::

            sage: UpperBounds.matching_bound(graphs.CompleteGraph(3))
            2

        """
        return g.num_verts() - matching_number(g)

    @staticmethod
    def fractional_alpha(g):
        r"""
        Compute the fractional independence number of the graph.

        EXAMPLES:

        ::

            sage: UpperBounds.fractional_alpha(graphs.CompleteGraph(3))
            1.5

        ::

            sage: UpperBounds.fractional_alpha(graphs.PathGraph(3))
            2.0

        """
        p = MixedIntegerLinearProgram(maximization=True)
        x = p.new_variable()
        p.set_objective(sum([x[v] for v in g.vertices()]))

        for v in g.vertices():
            p.add_constraint(x[v], max=1)

        for (u,v) in g.edge_iterator(labels=False):
            p.add_constraint(x[u] + x[v], max=1)

        return p.solve()

    @staticmethod
    def lovasz_theta(g):
        r"""
        Compute the value of the Lovasz theta function of the given graph.

        EXAMPLES:

        For an empty graph `G`, `\vartheta(G) = n`::

            sage: UpperBounds.lovasz_theta(Graph(2)) # rel tol 1e-3
            2.000

        For a complete graph `G`, `\vartheta(G) = 1`::

            sage: G = graphs.CompleteGraph(3)
            sage: UpperBounds.lovasz_theta(G) # rel tol 1e-3
            1.000

        For a pentagon (five-cycle) graph `G`, `\vartheta(G) = \sqrt{5}`::

            sage: G = graphs.CycleGraph(5)
            sage: UpperBounds.lovasz_theta(G) # rel tol 1e-3
            2.236

        For the Petersen graph `G`, `\vartheta(G) = 4`::

            sage: G = graphs.PetersenGraph()
            sage: UpperBounds.lovasz_theta(G) # rel tol 1e-3
            4.000

        """
        import cvxopt.base
        import cvxopt.solvers

        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = float(1e-10)
        cvxopt.solvers.options['reltol'] = float(1e-10)

        gc = g.complement()
        n = gc.num_verts()
        m = gc.num_edges()

        if n == 1:
            return 1.0

        d = m + n
        c = -1r * cvxopt.base.matrix([0.0r]*(n-1) + [2.0r]*(d-n))
        Xrow = [i*(1r+n) for i in xrange(n-1)] + [b+a*n for (a, b) in gc.edge_iterator(labels=False)]
        Xcol = range(n-1) + range(d-1)[n-1:]
        X = cvxopt.base.spmatrix(1.0r, Xrow, Xcol, (n*n, d-1r))

        for i in xrange(n-1):
            X[n*n-1r, i] = -1.0r

        sol = cvxopt.solvers.sdp(c, Gs=[-X], hs=[-cvxopt.base.matrix([0.0r]*(n*n-1r) + [-1.0r], (n,n))])
        v = 1.0r + cvxopt.base.matrix(-c, (1, d-1)) * sol['x']

        return v[0r]
