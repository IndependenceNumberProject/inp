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

    NOTES:

    The return value of this function may change depending on the functions
    included in the AlphaProperties, LowerBounds, and UpperBounds classes.

    AUTHORS:

    - Patrick Gaskill (2012-08-21)
    """
    n = 1
    count = 0

    while True:

        if verbose:
            print 'Graph{0} on {1} vert{2}'.format(['', 's'][n != 1],
                                                    n, ['ex', 'ices'][n != 1])
        gen = graphs.nauty_geng(str(n))
        while True:
            try:
                g = gen.next()
                count += 1

                if verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                if is_difficult(g):

                    if verbose:
                        print "\n\nFound a difficult graph!"

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
        lowerbounds_table += \
            "{0} & {1} \\\\\n".format(name, func(g)).replace('_', '\_')

    # Generate the latex for the upper bounds table
    upperbounds_table = ''
    for name, func in inspect.getmembers(UpperBounds, inspect.isfunction):
        upperbounds_table += \
            "{0} & {1} \\\\\n".format(name, func(g)).replace('_', '\_')

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

    INPUT:

    - ``g`` - sage.graphs.Graph -- the graph to be checked

    OUTPUT:

    - boolean -- return True if the input graph is considered difficult

    EXAMPLES:

    ::

        sage: G = Graph(1)
        sage: is_difficult(G)
        False

    NOTES:

    The return value of this function may change depending on the functions
    included in the AlphaProperties, LowerBounds, and UpperBounds classes.

    AUTHORS:

    - Patrick Gaskill (2012-08-21)
    """
    if has_alpha_property(g):
        return False

    lbound = lower_bound(g)
    ubound = upper_bound(g)

    if lbound == ubound:
        return False

    return True

def has_alpha_property(g):
    r"""
    This function determines if a given Graph `g` satisifes any of the known
    alpha-properties.

    INPUT:

    - ``g`` - sage.graphs.Graph -- the graph to be checked

    OUTPUT:

    - boolean - return True if the graph satisfies any alpha-properties

    EXAMPLES:

    ::

        sage: G = difficult_graph_search(verbose=False) # long time
        sage: has_alpha_property(G) # long time
        False

    NOTES:

    The return value of this function may change depending on the functions
    included in the AlphaProperties class.

    AUTHORS:

    - Patrick Gaskill (2012-08-21)
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

    INPUT:

    - ``g`` - sage.graphs.Graph -- the graph to be checked

    OUTPUT:

    - integer -- a lower bound for the independence number of the graph

    EXAMPLES:

    ::

        sage: G = Graph(1)
        sage: lower_bound(G)
        1

    ::

        sage: G = graphs.CompleteGraph(3)
        sage: lbound = lower_bound(G)
        sage: lbound >= 1 and lbound <= G.num_verts()
        True

    NOTES:

    The return value of this function may change depending on the functions
    included in the LowerBounds class.

    AUTHORS:

    - Patrick Gaskill (2012-08-21)
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

    INPUT:

    - ``g`` - sage.graphs.Graph -- the graph to be checked

    OUTPUT:

    - integer -- an upper bound for the independence number of the graph

   EXAMPLES:

    ::

        sage: G = Graph(1)
        sage: upper_bound(G)
        1

    ::

        sage: G = graphs.CompleteGraph(3)
        sage: ubound = upper_bound(G)
        sage: ubound >= 1 and ubound <= G.num_verts()
        True

    NOTES:

    The return value of this function may change depending on the functions
    included in the UpperBounds class.

    AUTHORS:

    - Patrick Gaskill (2012-08-21)
    """

    # The default upper bound is the number of vertices
    ubound = g.num_verts()

    # Loop through all the functions in UpperBounds class
    for name, func in inspect.getmembers(UpperBounds, inspect.isfunction):
        new_bound = func(g)
        if new_bound < ubound:
            ubound = new_bound
    return ubound

class AlphaProperties(object):
    @staticmethod
    def is_disconnected(g):
        r"""
        Determine if the graph is disconnected.

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - boolean -- True if the graph is disconnected

        EXAMPLES:

        ::

            sage: G = Graph(2)
            sage: AlphaProperties.is_disconnected(G)
            True

        ::

            sage: G = graphs.CompleteGraph(3)
            sage: AlphaProperties.is_disconnected(G)
            False

        NOTES:

        This property was added to solve the graph 'A?', or Graph(2).

        AUTHORS:

        - Patrick Gaskill (2012-08-21)
        """
        return not g.is_connected()

    @staticmethod
    def has_max_degree_order_minus_one(g):
        r"""
        Determine if the graph has a vertex with degree `n(G)-1`.

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - boolean -- True if the graph has a vertex with degree `n(G)-1`.

        EXAMPLES:

        ::

            sage: G = Graph(2)
            sage: AlphaProperties.has_max_degree_order_minus_one(G)
            False

        ::

            sage: G = graphs.CompleteGraph(3)
            sage: AlphaProperties.has_max_degree_order_minus_one(G)
            True

        NOTES:

        This property was added to solve the graph 'BW', or
        graphs.PathGraph(3).

        AUTHORS:

        - Patrick Gaskill (2012-08-21)
        """
        return max(g.degree()) == g.num_verts() - 1

    @staticmethod
    def is_claw_free(g):
        r"""
        Determine if the graph contains a claw, i.e., an induced `K_{1,3}`
        subgraph.

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - boolean -- True if the graph contains a claw

        EXAMPLES:

        ::

            sage: G = Graph(2)
            sage: AlphaProperties.is_claw_free(G)
            True

        ::

            sage: G = graphs.StarGraph(4)
            sage: AlphaProperties.is_claw_free(G)
            False

        NOTES:

        This property was added to solve the graph 'CU', or
        graphs.PathGraph(4).

        AUTHORS:

        - Patrick Gaskill (2012-08-21)
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

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - boolean -- True if the graph contains a pendant vertex

        EXAMPLES:

        ::

            sage: G = Graph(2)
            sage: AlphaProperties.has_pendant_vertex(G)
            False

        ::

            sage: G = graphs.StarGraph(4)
            sage: AlphaProperties.has_pendant_vertex(G)
            True

        NOTES:

        This property was added to solve the graph 'DCw'.

        AUTHORS:

        - Patrick Gaskill (2012-08-21)
        """
        return 1 in g.degree_sequence()

    @staticmethod
    def has_complete_closed_neighborhood(g):
        r"""
        Determine if the graph contains a complete closed neighborhood.

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - boolean -- True if the graph contains a complete closed neighborhood.

        EXAMPLES:

        ::

            sage: G = graphs.CycleGraph(4)
            sage: AlphaProperties.has_complete_closed_neighborhood(G)
            False

        ::

            sage: G = graphs.CompleteGraph(4)
            sage: AlphaProperties.has_complete_closed_neighborhood(G)
            True

        AUTHORS:

        - Patrick Gaskill (2012-08-22)
        """
        for v in g.vertices():
            if g.subgraph(g.neighbors(v)).is_clique():
                return True

        return False

    @staticmethod
    def is_bipartite(g):
        r"""
        Determine if the graph is bipartite.

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - boolean -- True if the graph is bipartite.

        EXAMPLES:

        ::

            sage: G = graphs.CompleteGraph(4)
            sage: AlphaProperties.is_bipartite(G)
            False

        ::

            sage: G = graphs.PathGraph(3)
            sage: AlphaProperties.is_bipartite(G)
            True

        NOTES:

        This property was added to solve the graph 'DFw'. This will eventually
        be made redundant by has_nonempty_KE_part() to be added later.

        AUTHORS:

        - Patrick Gaskill (2012-08-22)
        """
        return g.is_bipartite()

    @staticmethod
    def is_KE(g):
        r"""
        Determine if the graph is Konig-Egervary.

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - boolean -- True if the graph is Konig-Egervary.

        EXAMPLES:

        ::

            sage: G = graphs.PathGraph(3)
            sage: AlphaProperties.is_KE(G)
            True

        ::

            sage: G = graphs.CycleGraph(3)
            sage: AlphaProperties.is_KE(G)
            False

        AUTHORS:

        - Patrick Gaskill (2012-08-24)
        """
        alphaf = UpperBounds.fractional_alpha(g)
 
        # Ideally use_edge_labels=False but there is a bug in Sage
        if g.weighted():
            raise RuntimeWarning("There is a bug in sage, using the weighted matching number for now!")

        mu = int(g.matching(value_only=True, use_edge_labels=True))

        return alphaf + mu == g.num_verts()

class LowerBounds(object):
    @staticmethod
    def matching_number(g):
        r"""

        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - integer -- A lower bound on the independence number of the graph

        EXAMPLES:

        ::

            sage: G = graphs.CompleteGraph(3)
            sage: LowerBounds.matching_number(G)
            1

        NOTES:

        This property was added to solve the graph 'A_', or
        graphs.CompleteGraph(2).

        AUTHORS:

        - Patrick Gaskill (2012-08-21)
        """

        # Ideally use_edge_labels=False but there is a bug in Sage
        if g.weighted():
            raise RuntimeWarning("There is a bug in sage, using the weighted matching number for now!")

        return g.num_verts() - 2 * int(g.matching(value_only=True, use_edge_labels=True))

class UpperBounds(object):
    @staticmethod
    def matching_number(g):
        r"""
        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - integer -- An upper bound on the independence number of the graph

        ::

            sage: G = graphs.CompleteGraph(3)
            sage: UpperBounds.matching_number(G)
            2

        NOTES:

        This property was added to solve the graph 'A_', or
        graphs.CompleteGraph(2).

        AUTHORS:

        - Patrick Gaskill (2012-08-21)
        """

        # Ideally use_edge_labels=False but there is a bug in Sage
        if g.weighted():
            raise RuntimeWarning("There is a bug in sage, using the weighted matching number for now!")

        return g.num_verts() - int(g.matching(value_only=True))

    @staticmethod
    def fractional_alpha(g):
        r"""
        INPUT:

        - ``g`` - sage.graphs.Graph -- The graph to be checked

        OUTPUT:

        - float -- The fractional independence number of the graph

        ::

            sage: G = graphs.CompleteGraph(3)
            sage: UpperBounds.fractional_alpha(G)
            1.5

        ::

            sage: G = graphs.PathGraph(3)
            sage: UpperBounds.fractional_alpha(G)
            2.0

        AUTHORS:

        - Patrick Gaskill (2012-08-24)
        """
        p = MixedIntegerLinearProgram(maximization=True)
        x = p.new_variable()
        p.set_objective(sum([x[v] for v in g.vertices()]))

        for v in g.vertices():
            p.add_constraint(x[v] <= 1)

        for (u,v) in g.edge_iterator(labels=False):
            p.add_constraint(x[u] + x[v] <= 1)

        return p.solve()