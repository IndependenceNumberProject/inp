r"""
Search for difficult graphs in the Independence Number Project.

AUTHORS:

- Patrick Gaskill (2012-09-16): v0.2 refactored into INPGraph class

- Patrick Gaskill (2012-08-21): v0.1 initial version
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

import cvxopt.base
import cvxopt.solvers
import datetime
from string import Template
import os
import re
import subprocess
import sys
import time

# TODO: Include more functions from survey
# TODO: Get PDF exporting working again

from sage.all import Graph, graphs, Integer, Rational, floor, ceil, sqrt, \
                     MixedIntegerLinearProgram
from sage.misc.package import is_package_installed

class INPGraph(Graph):
    _nauty_count_pattern = re.compile(r'>Z (\d+) graphs generated')
    _save_path = os.path.expanduser("~/Dropbox/INP/")

    try:
        from progressbar import Bar, Counter, ETA, Percentage, ProgressBar
        _has_progressbar = True
    except ImportError:
        _has_progressbar = False

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)

    @classmethod
    def survey(cls, func, order):
        # TODO: Write documentation
        # TODO: Is it possible to write tests for this?
        if not is_package_installed("nauty"):
            raise TypeError, "The nauty package is required to survey a bound or property."

        sys.stdout.write("Counting graphs of order {0}... ".format(order))
        sys.stdout.flush()
        num_graphs_to_check = cls.count_viable_graphs(order)
        print num_graphs_to_check

        if _has_progressbar:
            pbar = ProgressBar(widgets=["Testing: ", Counter(), Bar(), ETA()], maxval=num_graphs_to_check, fd=sys.stdout).start()

        gen = graphs.nauty_geng("-cd3D{0} {1}".format(order-2, order))
        counter = 0
        hits = 0

        is_alpha_property = hasattr(func, '_is_alpha_property') and func._is_alpha_property
        is_bound = (hasattr(func, '_is_lower_bound') and func._is_lower_bound) or \
                   (hasattr(func, '_is_upper_bound') and func._is_upper_bound)

        while True:
            try:
                g = INPGraph(gen.next())

                if is_alpha_property:
                    if func(g):
                        hits += 1
                elif is_bound:
                    if func(g) == g.independence_number():
                        hits += 1

                counter += 1

                if _has_progressbar:
                    pbar.update(counter)
                    sys.stdout.flush()

            except StopIteration:
                if _has_progressbar:
                    pbar.finish()

                if is_alpha_property:
                    print "{0} out of {1} graphs of order {2} satisfied {3}.".format(hits, counter, order, func.__name__)
                elif is_bound:
                    print "{0} out of {1} graphs of order {2} were predicted by {3}.".format(hits, counter, order, func.__name__)
                return

            except KeyboardInterrupt:
                print "\nStopped."
                return

    @classmethod
    def count_viable_graphs(cls, order):
        # TODO: Write documentation
        # TODO: Write tests
        if not is_package_installed("nauty"): 
            raise TypeError, "The nauty package is required to count viable graphs."

        # Graphs with < 6 vertices will have pendant or foldable vertices.
        if order < 6:
            return 0

        output = subprocess.check_output(["{0}/local/bin/nauty-geng".format(SAGE_ROOT),
                                 "-cud3D{0}".format(order-2), str(order)], stderr=subprocess.STDOUT)
        m = cls._nauty_count_pattern.search(output)
        return int(m.group(1))

    @classmethod
    def _next_difficult_graph_of_order(cls, order, verbose=True):
        if not is_package_installed("nauty"): 
            raise TypeError, "The nauty package is required to find difficult graphs."

        # Graphs with < 6 vertices will have pendant or foldable vertices.
        if order < 6:
            raise ValueError, "There are no difficult graphs with less than 6 vertices."

        if verbose:
            sys.stdout.write("Counting graphs of order {0}... ".format(order))
            sys.stdout.flush()
            num_graphs_to_check = cls.count_viable_graphs(order)
            print num_graphs_to_check

            if _has_progressbar:
                pbar = ProgressBar(widgets=["Testing: ", Counter(), Bar(), ETA()], maxval=num_graphs_to_check, fd=sys.stdout).start()

        gen = graphs.nauty_geng("-cd3D{0} {1}".format(order-2, order))
        counter = 0

        while True:
            try:
                g = INPGraph(gen.next())

                if g.is_difficult():
                    if verbose:
                        if _has_progressbar:
                            pbar.finish()
                        print "Found a difficult graph: {0}".format(g.graph6_string())
                        g.save_files()
                    return g

                counter += 1

                if verbose and _has_progressbar:
                    pbar.update(counter)
                    sys.stdout.flush()

            except StopIteration:
                if verbose and _has_progressbar:
                    pbar.finish()

                return None
        
    @classmethod
    def next_difficult_graph(cls, order=None, verbose=True, write_to_pdf=False):
        # TODO: Is it possible to write good tests for this?
        # TODO: Write this function including a progress bar
        # pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=300).start()
        # for i in range(300):
        #     #time.sleep(0.01)
        #     pbar.update(i+1)
        # pbar.finish()
        r"""
        This function returns the smallest graph considered difficult by INP theory.

        INPUT:

        - ``verbose`` - boolean -- Print progress to the console and save graph
            information as a dossier PDF and a PNG image.

        NOTES:

        The return value of this function may change depending on the functions
        included in the _lower_bounds, _upper_bounds, and _alpha_properties
        settings.
        """
        if not is_package_installed("nauty"): 
            raise TypeError, "The nauty package is not required to find difficult graphs."

        # Graphs with < 6 vertices will have pendant or foldable vertices.

        if order is None:
            n = 6
        else:
            if order < 6:
                raise ValueError, "There are no difficult graphs with less than 6 vertices."

            n = order

        while True:
            try:
                g = cls._next_difficult_graph_of_order(n, verbose)
                if g is None:
                    n += 1
                else:
                    return g
            except KeyboardInterrupt:
                if verbose:
                    sys.stdout.flush()
                    print "\nStopped."
                return None

    def is_difficult(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function determines if the graph is difficult as described by
        INP theory.

        NOTES:

        The return value of this function may change depending on the functions
        included in the _lower_bounds, _upper_bounds, and _alpha_properties
        settings.
        """
        if self.has_alpha_property():
            return False

        lbound = ceil(self.best_lower_bound())
        ubound = floor(self.best_upper_bound())

        if lbound == ubound:
            return False

        return True

    def best_lower_bound(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function computes a lower bound for the independence number of the
        graph.

        NOTES:

        The return value of this function may change depending on the functions
        included in the _lower_bounds setting.
        """
        # The default bound is 1
        lbound = 1

        for func in self._lower_bounds:
            try:
                new_bound = func(self)
                if new_bound > lbound:
                    lbound = new_bound
            except ValueError:
                pass

        return lbound

    def best_upper_bound(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function computes an upper bound for the independence number of
        the graph.

        NOTES:

        The return value of this function may change depending on the functions
        included in the _upper_bounds setting.
        """
        # The default upper bound is the number of vertices
        ubound = self.order()

        for func in self._upper_bounds:
            try:
                new_bound = func(self)
                if new_bound < ubound:
                    ubound = new_bound
            except ValueError:
                pass

        return ubound

    def has_alpha_property(self):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function determines if the graph satisifes any of the known
        alpha-properties or alpha-reductions.

        NOTES:

        The return value of this function may change depending on the functions
        included in the _alpha_properties setting.
        """
        for func in self._alpha_properties:
            try:
                if func(self):
                    return True
            except ValueError:
                pass

        return False

    def save_files(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = "difficult_graph_{0}".format(timestamp)
        folder_path = "{0}/{1}".format(self._save_path, filename)

        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except IOError:
            "Can't make directory {0}".format(folder_path)

        try:
            self.plot().save("{0}/{1}.png".format(folder_path, filename))
            print "Plot saved."
        except IOError:
            print "Couldn't save {0}.png".format(filename)

        try:
            self._export_pdf(folder_path, filename)
            print "Dossier saved."
        except IOError:
            print "Couldn't save {0}.pdf".format(filename)

    def _export_pdf(self, folder_path, filename):
        r"""
        Generate the latex for the information box.
        """
        info_table = """
        \\rowcolor{{LightGray}} $n$ & {0} \\\\
        \\rowcolor{{LightGray}} $e$ & {1} \\\\
        \\rowcolor{{LightGray}} $\\alpha$ & {2} \\\\
        """.format(self.order(), self.size(), self.independence_number())

        # Generate the latex for the lower bounds table
        lowerbounds_table = ''
        for func in self._lower_bounds:
            name = func.__name__
            value = func(g)

            try:
                if value in ZZ:
                    lowerbounds_table += \
                        "{0} & {1} \\\\\n".format(name, int(value)).replace('_', r'\_')
                else:
                    lowerbounds_table += \
                       "{0} & {1:.3f} \\\\\n".format(name, float(value)).replace('_', r'\_')
            except (AttributeError, ValueError):
                print "Can't format", name, value, "for LaTeX output."
                lowerbounds_table += \
                    "{0} & {1} \\\\\n".format(name, '?').replace('_', r'\_')

        # Generate the latex for the upper bounds table
        upperbounds_table = ''
        for func in self._upper_bounds:
            name = func.__name__
            value = func(g)

            try:
                if value in ZZ:
                    upperbounds_table += \
                        "{0} & {1} \\\\\n".format(name, int(value)).replace('_', r'\_')
                else:
                    upperbounds_table += \
                        "{0} & {1:.3f} \\\\\n".format(name, float(value)).replace('_', r'\_')
            except (AttributeError, ValueError):
                print "Can't format", name, value, "for LaTeX output."
                upperbounds_table += \
                    "{0} & {1} \\\\\n".format(name, '?').replace('_', r'\_')

        # Generate the latex for the alpha properties table
        alphaproperties_table = ''
        for func in self._alpha_properties:
            name = func.__name__
            alphaproperties_table += \
                "{0} \\\\\n".format(name).replace('_', r'\_').replace('~', r'{\textasciitilde}')

        # Insert all the generated latex into the template file
        template_file = open('dossier_template.tex', 'r')
        template = template_file.read()
        s = Template(template)

        output = s.substitute(graph=latex(self), 
                              name=self.graph6_string().replace('_', '\_'),
                              info=info_table,
                              lowerbounds=lowerbounds_table, 
                              upperbounds=upperbounds_table,
                              alphaproperties=alphaproperties_table)
        latex_filename = "{0}/{1}.tex".format(folder_path, filename)

        # Write the latex to a file then run pdflatex on it
        # TODO: Handle calling pdflatex and its errors better.
        try:
            latex_file = open(latex_filename, 'w')
            latex_file.write(output)
            latex_file.close()
            with open(os.devnull, 'wb') as devnull:
                subprocess.call(['/usr/texbin/pdflatex', '-output-directory',
                    folder_path, latex_filename],
                    stdout=devnull, stderr=subprocess.STDOUT)
        except:
            pass

    def matching_number(self):
        # TODO: This needs to be updated when Sage 5.3 is released.
        r"""
        Compute the traditional matching number `\mu`.

        EXAMPLES:

        ::
            sage: tests = [2, graphs.CompleteGraph(3), graphs.PathGraph(3), \
                           graphs.StarGraph(3), 'EXCO', graphs.CycleGraph(5), \
                           graphs.PetersenGraph()]
            sage: [INPGraph(g).matching_number() for g in tests]
            [0, 1, 1, 1, 2, 2, 5]

        WARNINGS:

        Ideally we would set use_edge_labels=False to ignore edge weighting and
        always compute the traditional matching number, but there is a bug
        in Sage 5.2 that returns double this number. Calling this on an
        edge-weighted graph will NOT give the usual matching number.
        """
        return int(self.matching(value_only=True))

    mu = matching_number

    def independence_number(self):
        r"""
        Compute the independence number using the Sage built-in independent_set
        method. This uses the Cliquer algorithm, which does not run in
        polynomial time.

        EXAMPLES:

        ::
            sage: tests = [2, graphs.CompleteGraph(3), graphs.PathGraph(3), \
                           graphs.StarGraph(3), 'EXCO', graphs.CycleGraph(5), \
                           graphs.PetersenGraph()]
            sage: [INPGraph(g).independence_number() for g in tests]
            [2, 1, 2, 3, 4, 2, 4]

        """
        return int(len(self.independent_set()))

    alpha = independence_number

    def bipartite_double_cover(self):
        r"""
        Return a bipartite double cover of the graph, also known as the
        bidouble.

        EXAMPLES:

        ::
            sage: b = INPGraph(2).bipartite_double_cover()
            sage: b.is_isomorphic(Graph(4))
            True

        ::
            sage: b = INPGraph(graphs.CompleteGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(graphs.CycleGraph(6))
            True

        ::
            sage: b = INPGraph(graphs.PathGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('EgCG'))
            True

        ::
            sage: b = INPGraph(graphs.StarGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('Gs?GGG'))
            True

        ::
            sage: b = INPGraph('EXCO').bipartite_double_cover()
            sage: b.is_isomorphic(Graph('KXCO?C@??A_@'))
            True

        ::
            sage: b = INPGraph(graphs.CycleGraph(5)).bipartite_double_cover()
            sage: b.is_isomorphic(graphs.CycleGraph(10))
            True

        ::
            sage: b = INPGraph(graphs.PetersenGraph()).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('SKC_GP@_a?O?C?G??__OO?POAI??a_@D?'))
            True
        """
        return INPGraph(self.tensor_product(graphs.CompleteGraph(2)))

    bidouble = bipartite_double_cover
    kronecker_double_cover = bipartite_double_cover
    canonical_double_cover = bipartite_double_cover

    def closed_neighborhood(self, verts):
        # TODO: Write tests
        if isinstance(verts, list):
            neighborhood = []
            for v in verts:
                neighborhood += [v] + self.neighbors(v)
            return list(set(neighborhood))
        else:
            return [verts] + self.neighbors(verts)

    def open_neighborhood(self, verts):
        # TODO: Write tests
        if isinstance(verts, list):
            neighborhood = []
            for v in verts:
                neighborhood += self.neighbors(v)
            return list(set(neighborhood))
        else:
            return self.neighbors(verts)

    def max_degree(self):
        # TODO: Write tests
        return max(self.degree())

    def min_degree(self):
        # TODO: Write tests
        return min(self.degree())

    def union_MCIS(self):
        # TODO: Write more tests
        r"""
        Return a union of maximum critical independent sets (MCIS).

        EXAMPLES:

        ::
            sage: G = INPGraph('Cx')
            sage: G.union_MCIS()
            [0, 1, 3]

        ::
            sage: G = INPGraph(graphs.CycleGraph(4))
            sage: G.union_MCIS()
            [0, 1, 2, 3]

        """
        b = self.bipartite_double_cover()
        alpha = b.order() - b.matching_number()

        result = []

        for v in self.vertices():
            test = b.copy()
            test.delete_vertices(b.closed_neighborhood([(v,0), (v,1)]))
            alpha_test = test.order() - test.matching_number() + 2
            if alpha_test == alpha:
                result.append(v)

        return result

    ###########################################################################
    # Alpha properties
    ###########################################################################

    def has_max_degree_order_minus_one(self):
        # TODO: Write tests
        return self.max_degree() == self.order() - 1
    has_max_degree_order_minus_one._is_alpha_property = True

    def is_claw_free(self):
        # TODO: Write tests
        subsets = combinations_iterator(self.vertices(), 4)
        for subset in subsets:
            if self.subgraph(subset).degree_sequence() == [3,1,1,1]:
                return False
        return True
    is_claw_free._is_alpha_property = True

    def has_pendant_vertex(self):
        return 1 in self.degree()
    has_pendant_vertex._is_alpha_property = True

    def has_simplicial_vertex(self):
        # TODO: Write tests
        for v in self.vertices():
            if self.subgraph(self.neighbors(v)).is_clique():
                return True

        return False
    has_simplicial_vertex._is_alpha_property = True

    def is_KE(self):
        # TODO: Write tests
        c = self.union_MCIS()
        nc = []
        for v in c:
            nc.extend(self.neighbors(v))

        return list(set(c + nc)) == self.vertices()
    is_KE._is_alpha_property = True

    def is_almost_KE(self):
        # TODO: Write tests
        subsets = combinations_iterator(self.vertices(), self.order() - 1)
        for subset in subsets:
            if self.subgraph(subset).is_KE():
                return True

        return False
    is_almost_KE._is_alpha_property = True

    def has_nonempty_KE_part(self):
        # TODO: Write tests
        if self.union_MCIS():
            return True
        else:
            return False
    has_nonempty_KE_part._is_alpha_property = True

    def is_foldable(self):
        # TODO: Write tests
        # TODO: Write this function
        pass
    is_foldable._is_alpha_property = True

    ###########################################################################
    # Lower bounds
    ###########################################################################

    def matching_lower_bound(self):
        # TODO: Write better tests
        r"""
        Compute the matching number lower bound.

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.matching_lower_bound()
            1

        """
        return self.order() - 2 * self.matching_number()
    matching_lower_bound._is_lower_bound = True

    def residue(self):
        # TODO: Write tests
        seq = self.degree_sequence()

        while seq[0] > 0:
            d = seq.pop(0)
            seq[:d] = [k-1 for k in seq[:d]]
            seq.sort(reverse=True)

        return len(seq)
    residue._is_lower_bound = True

    def average_degree_bound(self):
        # TODO: Write tests
        n = Integer(self.order())
        d = Rational(self.average_degree())
        return n / (1 + d)
    average_degree_bound._is_lower_bound = True

    def caro_wei(self):
        # TODO: Write better tests
        r"""

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.caro_wei()
            1

        ::

            sage: G = INPGraph(graphs.PathGraph(3))
            sage: G.caro_wei()
            4/3

        """
        return sum([1/(1+Integer(d)) for d in self.degree()])
    caro_wei._is_lower_bound = True

    def wilf(self):
        # TODO: Write tests
        n = Integer(self.order())
        max_eigenvalue = max(self.adjacency_matrix().eigenvalues())
        return n / (1 + max_eigenvalue)
    wilf._is_lower_bound = True

    def hansen_zheng_lower_bound(self):
        # TODO: Write tests
        n = Integer(self.order())
        e = Integer(self.size())
        return ceil(n - (2 * e)/(1 + floor(2 * e / n)))
    hansen_zheng_lower_bound._is_lower_bound = True

    def harant(self):
        # TODO: Write tests
        n = Integer(self.order())
        e = Integer(self.size())
        term = 2 * e + n + 1
        return (1/2) * (term - sqrt(term^2 - 4*n^2))
    harant._is_lower_bound = True

    ###########################################################################
    # Upper bounds
    ###########################################################################

    def matching_upper_bound(self):
        # TODO: Write better tests
        r"""
        Compute the matching number upper bound.

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.matching_upper_bound()
            2

        """
        return self.order() - self.matching_number()
    matching_upper_bound._is_upper_bound = True

    def fractional_alpha(self):
        # TODO: Write better tests
        r"""
        Compute the fractional independence number of the graph.

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.fractional_alpha()
            1.5

        ::

            sage: G = INPGraph(graphs.PathGraph(3))
            sage: G.fractional_alpha()
            2.0

        """
        p = MixedIntegerLinearProgram(maximization=True)
        x = p.new_variable()
        p.set_objective(sum([x[v] for v in self.vertices()]))

        for v in self.vertices():
            p.add_constraint(x[v], max=1)

        for (u,v) in self.edge_iterator(labels=False):
            p.add_constraint(x[u] + x[v], max=1)

        return p.solve()
    fractional_alpha._is_upper_bound = True

    def lovasz_theta(self):
        # TODO: Write better tests
        # TODO: There has to be a nicer way of doing this.
        r"""
        Compute the value of the Lovasz theta function of the given graph.

        EXAMPLES:

        For an empty graph `G`, `\vartheta(G) = n`::

            sage: G = INPGraph(2)
            sage: G.lovasz_theta()
            2.0

        For a complete graph `G`, `\vartheta(G) = 1`::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.lovasz_theta()
            1.0

        For a pentagon (five-cycle) graph `G`, `\vartheta(G) = \sqrt{5}`::

            sage: G = INPGraph(graphs.CycleGraph(5))
            sage: G.lovasz_theta()
            2.236

        For the Petersen graph `G`, `\vartheta(G) = 4`::

            sage: G = INPGraph(graphs.PetersenGraph())
            sage: G.lovasz_theta()
            4.0
        """
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['abstol'] = float(1e-10)
        cvxopt.solvers.options['reltol'] = float(1e-10)

        gc = self.complement()
        n = gc.order()
        m = gc.size()

        if n == 1:
            return 1.0

        d = m + n
        c = -1 * cvxopt.base.matrix([0.0]*(n-1) + [2.0]*(d-n))
        Xrow = [i*(1+n) for i in xrange(n-1)] + [b+a*n for (a, b) in gc.edge_iterator(labels=False)]
        Xcol = range(n-1) + range(d-1)[n-1:]
        X = cvxopt.base.spmatrix(1.0, Xrow, Xcol, (n*n, d-1))

        for i in xrange(n-1):
            X[n*n-1, i] = -1.0

        sol = cvxopt.solvers.sdp(c, Gs=[-X], hs=[-cvxopt.base.matrix([0.0]*(n*n-1) + [-1.0], (n,n))])
        v = 1.0 + cvxopt.base.matrix(-c, (1, d-1)) * sol['x']

        return round(v[0], 3)
    lovasz_theta._is_upper_bound = True

    def kwok(self):
        # TODO: Write better tests
        r"""
        Compute the upper bound `\alpha \leq n - \frac{e}{\Delta}` that is
        credited to Kwok, or possibly "folklore."

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.kwok()
            3/2

        ::

            sage: G = INPGraph(graphs.PathGraph(3))
            sage: G.kwok()
            2

        """
        n = Integer(self.order())
        e = Integer(self.size())
        Delta = Integer(self.max_degree())

        if Delta == 0:
            raise ValueError("Kwok bound is not defined for graphs with maximum degree 0.")

        return n - e / Delta
    kwok._is_upper_bound = True

    def hansen_zheng_upper_bound(self):
        # TODO: Write better tests
        r"""
        Compute an upper bound `\frac{1}{2} + \sqrt{\frac{1/4} + n^2 - n - 2e}` 
        given by Hansen and Zheng, 1993.

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.hansen_zheng_upper_bound()
            1

        """
        n = Integer(self.order())
        e = Integer(self.size())
        return floor(.5 + sqrt(.25 + n**2 - n - 2*e))
    hansen_zheng_upper_bound._is_upper_bound = True

    def min_degree_bound(self):
        r"""
        Compute the upper bound `\alpha \leq n - \delta`. This bound probably
        belong to "folklore."

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.min_degree_bound()
            1

        ::

            sage: G = INPGraph(graphs.PathGraph(4))
            sage: G.min_degree_bound()
            3

        """
        return self.order() - self.min_degree()
    min_degree_bound._is_upper_bound = True

    def cvetkovic(self):
        # TODO: Write better tests
        r"""
        Compute the Cvetkovic bound `\alpha \leq p_0 + min\{p_-, p_+\}`, where
        `p_-, p_0, p_+` denote the negative, zero, and positive eigenvalues 
        of the adjacency matrix of the graph respectively.

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.PetersenGraph())
            sage: G.cvetkovic()
            4

        """
        eigenvalues = self.adjacency_matrix().eigenvalues()
        [positive, negative, zero] = [0, 0, 0]
        for e in eigenvalues:
            if e > 0:
                positive += 1
            elif e < 0:
                negative += 1
            else:
                zero += 1

        return zero + min([positive, negative])
    cvetkovic._is_upper_bound = True

    def annihilation_number(self):
        # TODO: Write better tests
        r"""
        Compute the annhilation number of the graph.

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.annihilation_number()
            2

        ::

            sage: G = INPGraph(graphs.StarGraph(3))
            sage: G.annihilation_number()
            4

        """
        seq = sorted(self.degree())
        n = self.order()

        a = 1
        while sum(seq[:a]) <= sum(seq[a:]):
            a += 1

        return a
    annihilation_number._is_upper_bound = True

    def borg(self):
        # TODO: Write better tests
        r"""
        Compute the upper bound given by Borg.

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.borg()
            2

        """
        n = Integer(self.order())
        Delta = Integer(self.max_degree())

        if Delta == 0:
            raise ValueError("Borg bound is not defined for graphs with maximum degree 0.")

        return n - ceil((n-1) / Delta)
    borg._is_upper_bound = True

    def cut_vertices_bound(self):
        r"""

        EXAMPLES:

        ::

            sage: G = INPGraph(graphs.PathGraph(5))
            sage: G.cut_vertices_bound()
            3

        """
        n = Integer(self.order())
        C = Integer(len(self.blocks_and_cut_vertices()[1]))
        return n - C/2 - Integer(1)/2
    cut_vertices_bound._is_upper_bound = True

    _alpha_properties = [is_claw_free, has_simplicial_vertex, is_KE, is_almost_KE, has_nonempty_KE_part]
    _lower_bounds = [matching_lower_bound, residue, average_degree_bound, caro_wei, wilf, hansen_zheng_lower_bound, harant]
    _upper_bounds = [matching_upper_bound, fractional_alpha, lovasz_theta, kwok, hansen_zheng_upper_bound, min_degree_bound, cvetkovic, annihilation_number, borg, cut_vertices_bound]
