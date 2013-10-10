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
from functools import wraps
from string import Template
from itertools import imap
import os
import re
import subprocess
import sys
import time
import warnings

from sage.graphs.graph import Graph
from sage.graphs.graph_generators import graphs
from sage.rings.integer import Integer
from sage.rings.rational import Rational
from sage.functions.other import floor, ceil, sqrt
from sage.numerical.mip import MixedIntegerLinearProgram
from sage.misc.package import is_package_installed
from sage.rings.finite_rings.integer_mod import Mod
import sage.version
#from sage.combinat.combinat import Combinations
from sage.combinat.matrices.latin import LatinSquare
from sage.matrix.all import matrix
from sage.matrix.matrix_integer_dense import Matrix_integer_dense

# TODO: Include more functions from survey

try:
    from progressbar import Bar, Counter, ETA, Percentage, ProgressBar
    _INPGraph__has_progressbar = True
except ImportError:
    _INPGraph__has_progressbar = False

class INPGraph(Graph):
    _nauty_count_pattern = re.compile(r'>Z (\d+) graphs generated')
    _save_path = os.path.expanduser("~/Dropbox/INP")

    def memoize_graphs(func):
        func._cache = {}
        @wraps(func)
        def memo(g):
            key = g.graph6_string()
            if key not in func._cache:
                func._cache[key] = func(g)
            return func._cache[key]
        return memo

    def __init__(self, *args, **kwargs):
        Graph.__init__(self, *args, **kwargs)

    @classmethod
    def survey(cls, func, order):
        # TODO: Write documentation
        # TODO: Is it possible to write tests for this?
        if not is_package_installed("nauty"):
            raise TypeError, "The nauty package is required to survey a bound or property."

        # Graphs with < 6 vertices will have pendant or foldable vertices.
        if order < 6:
            raise ValueError, "There are no difficult graphs with less than 6 vertices."

        sys.stdout.write("Counting graphs of order {0}... ".format(order))
        sys.stdout.flush()
        num_graphs_to_check = cls.count_viable_graphs(order)
        print num_graphs_to_check

        if __has_progressbar:
            pbar = ProgressBar(widgets=["Testing: ", Counter(), Bar(), ETA()], maxval=num_graphs_to_check, fd=sys.stdout).start()
        
        gen = graphs.nauty_geng("-cd3D{0} {1}".format(order-2, order))
        counter = 0
        hits = 0

        is_alpha_property = hasattr(func, '_is_alpha_property') and func._is_alpha_property
        is_lower_bound = hasattr(func, '_is_lower_bound') and func._is_lower_bound
        is_upper_bound = hasattr(func, '_is_upper_bound') and func._is_upper_bound

        while True:
            try:
                g = INPGraph(gen.next())

                try:
                    if is_alpha_property:
                        if func(g):
                            hits += 1
                    elif is_lower_bound:
                        if ceil(func(g)) == g.independence_number():
                            hits += 1
                    elif is_upper_bound:
                        if floor(func(g)) == g.independence_number():
                            hits += 1

                except ValueError:
                    pass

                counter += 1

                if __has_progressbar:
                    pbar.update(counter)
                else:
                    sys.stdout.write("Testing order {0}: {1}/{2} ({3:.2f}%)\r".format(order, counter, num_graphs_to_check, (float(counter)/num_graphs_to_check)*100))    
                sys.stdout.flush()

            except StopIteration:
                if __has_progressbar:
                    pbar.finish()

                if is_alpha_property:
                    print "{0} out of {1} graphs of order {2} satisfied {3}.".format(hits, counter, order, func.__name__)
                elif is_lower_bound or is_upper_bound:
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
    def _next_difficult_graph_of_order(cls, order, verbose=True, save=False):
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

            if __has_progressbar:
                pbar = ProgressBar(widgets=["Testing: ", Counter(), Bar(), ETA()], maxval=num_graphs_to_check, fd=sys.stdout).start()

        gen = graphs.nauty_geng("-cd3D{0} {1}".format(order-2, order))
        counter = 0

        while True:
            try:
                g = INPGraph(gen.next())
                
                if g.is_difficult():
                    if verbose:
                        if __has_progressbar:
                            pbar.finish()
                        print "Found a difficult graph: {0} (Checked {1}/{2} graphs of order {3}.)".format(g.graph6_string(), counter, num_graphs_to_check, order)

                    if save:
                        g.save_files()

                    return g

                counter += 1

                if verbose:
                    if __has_progressbar:
                        pbar.update(counter)
                    else:
                        sys.stdout.write("Testing order {0}: {1}/{2} ({3:.2f}%)\r".format(order, counter, num_graphs_to_check, (float(counter)/num_graphs_to_check)*100))
                    sys.stdout.flush()

            except StopIteration:
                if verbose:
                    if __has_progressbar:
                        pbar.finish()
                    else:
                        print

                    print "No difficult graphs found."

                return None

    @classmethod
    def next_difficult_graph(cls, order=None, verbose=True, save=False):
        # TODO: Is it possible to write good tests for this?
        r"""
        This function returns the smallest graph considered difficult by INP theory.

        INPUT:

        - ``order`` - int -- Begin checking for difficult graphs at the given order.

        - ``verbose`` - boolean -- Print progress to the console.

        - ``save`` - boolean -- Save a PDF and PNG image of the difficult graph that is found.

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
                g = cls._next_difficult_graph_of_order(n, verbose, save)
                if g is None:
                    n += 1
                else:
                    return g
            except KeyboardInterrupt:
                if verbose:
                    sys.stdout.flush()
                    print "\nStopped."
                return None

    @classmethod
    def find_example(cls, func, order=1, verbose=True):
        r"""
        Returns the first connected graph that satisfies the given function.
        """
        if not is_package_installed("nauty"): 
            raise TypeError, "The nauty package is required to find graphs."

        while True:
            try:
                output = subprocess.check_output(["{0}/local/bin/nauty-geng".format(SAGE_ROOT),
                         "-cu", str(order)], stderr=subprocess.STDOUT)
                m = cls._nauty_count_pattern.search(output)
                num_graphs_to_check = int(m.group(1))
        
                if verbose:
                    sys.stdout.write("Counting graphs of order {0}... ".format(order))
                    sys.stdout.flush()
                    print num_graphs_to_check

                    if __has_progressbar:
                        pbar = ProgressBar(widgets=["Testing: ", Counter(), Bar(), ETA()], maxval=num_graphs_to_check, fd=sys.stdout).start()

                gen = graphs.nauty_geng("-c {0}".format(order))
                counter = 0

                while True:
                    try:
                        g = INPGraph(gen.next())
                        
                        if func(g):
                            if verbose:
                                if __has_progressbar:
                                    pbar.finish()
                                print "Found an example graph: {0} (Checked {1}/{2} graphs of order {3}.)".format(g.graph6_string(), counter, num_graphs_to_check, order)

                            g.show()
                            return g

                        counter += 1

                        if verbose:
                            if __has_progressbar:
                                pbar.update(counter)
                            else:
                                sys.stdout.write("Testing order {0}: {1}/{2} ({3:.2f}%)\r".format(order, counter, num_graphs_to_check, (float(counter)/num_graphs_to_check)*100))
                            sys.stdout.flush()

                    except StopIteration:
                        if verbose:
                            if __has_progressbar:
                                pbar.finish()
                            else:
                                print
                            print "No example graphs found."
                        
                        order += 1
                        break

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
        # TODO: Write documentation
        # TODO: Is it possible to write good tests for this?
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = "difficult_graph_{0}".format(timestamp)
        folder_path = "{0}/{1}".format(self._save_path, filename)

        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except IOError:
            "Can't make directory {0}".format(folder_path)

        (saved_plot, saved_pdf) = (False, False)

        try:
            self.plot().save("{0}/{1}.png".format(folder_path, filename))
            #print "Plot saved to {0}{1}.png".format(folder_path, filename)
            saved_plot = True
        except IOError:
            print "Couldn't save {0}/{1}.png".format(folder_path, filename)

        try:
            self._export_pdf(folder_path, filename)
            #print "Dossier saved to {0}{1}.pdf".format(folder_path, filename)
            saved_pdf = True
        except IOError:
            print "Couldn't save {0}/{1}.pdf".format(folder_path, filename)

        if saved_plot or saved_pdf:
            print "Saved graph information to: \n  {0}".format(folder_path)

    def _export_pdf(self, folder_path, filename):
        # TODO: Write documentation
        # TODO: Is it possible to write good tests for this?
        # TODO: Check for tkz style files

        if self.is_difficult():
            difficult_text = "\\textbf{This graph is difficult!} & \\danger \\\\"
        else:
            difficult_text = ""
        
        # Generate the latex for the alpha properties table
        alphaproperties = {}
        for func in self._alpha_properties:
            name = func.__name__
            try:
                if func(self):
                    print_value = "\ding{51}"
                else:
                    print_value = "\ding{56}"
            except ValueError:
                print_value = "$\\varnothing$"
            alphaproperties[name] = print_value

        # Sort by name ascending
        alphaproperties_keys_sorted = sorted(alphaproperties.keys())
        alphaproperties_table = ''

        for name in alphaproperties_keys_sorted:
            print_value = alphaproperties[name]
            alphaproperties_table += \
                "{0} & {1} \\\\\n".format(self._latex_escape(name), print_value)

        # Generate the latex for the lower bounds table
        lowerbounds = {}
        for func in self._lower_bounds:
            name = func.__name__
            try:
                sort_value = func(self)
                if sort_value in ZZ:
                    print_value = Integer(sort_value).str()
                elif sort_value in RR:
                    print_value = "{0:.3f}".format(float(sort_value))
                else:
                    print_value = self._latex_escape(str(sort_value))
            except ValueError:
                sort_value = -1
                print_value = "$\\varnothing$"
            lowerbounds[name] = (sort_value, print_value)

        # Sort by sort_value ascending, then by name ascending
        lowerbounds_keys_sorted = sorted(lowerbounds.keys(), cmp=lambda a,b: cmp((lowerbounds[a][0], a), (lowerbounds[b][0], b)))
        lowerbounds_table = ''

        for name in lowerbounds_keys_sorted:
            sort_value, print_value = lowerbounds[name]
            try:
                 lowerbounds_table += \
                     "{0} & {1} \\\\\n".format(self._latex_escape(name), print_value)

            except (AttributeError, ValueError):
                print "Can't format", name, value, "for LaTeX output."
                lowerbounds_table += \
                    "{0} & {1} \\\\\n".format(self._latex_escape(name), '?')

        # Generate the latex for the upper bounds table
        upperbounds = {}
        for func in self._upper_bounds:
            name = func.__name__
            try:
                sort_value = func(self)
                if sort_value in ZZ:
                    print_value = Integer(sort_value).str()
                elif sort_value in RR:
                    print_value = "{0:.3f}".format(float(sort_value))
                else:
                    print_value = self._latex_escape(str(sort_value))

            except ValueError:
                sort_value = sys.maxint
                print_value = "$\\varnothing$"
            upperbounds[name] = (sort_value, print_value)

        # Sort by sort_value ascending, then by name ascending
        upperbounds_keys_sorted = sorted(upperbounds.keys(), cmp=lambda a,b: cmp((upperbounds[a][0], a), (upperbounds[b][0], b)))
        upperbounds_table = ''

        for name in upperbounds_keys_sorted:
            sort_value, print_value = upperbounds[name]
            try:
                upperbounds_table += \
                     "{0} & {1} \\\\\n".format(self._latex_escape(name), print_value)

            except (AttributeError, ValueError):
                print "Can't format", name, value, "for LaTeX output."
                upperbounds_table += \
                    "{0} & {1} \\\\\n".format(self._latex_escape(name), '?')


        # Insert all the generated latex into the template file
        template_file = open('template.tex', 'r')
        template = template_file.read()
        s = Template(template)

        # Want to use the circular embedding and Dijkstra style for the PDF,
        # but we'll set it back to whatever the user had after we're done.

        opts = self.latex_options()
        old_style = opts.get_option('tkz_style')
        opts.set_option('tkz_style', 'Dijkstra')
        default_tikz_latex = latex(self)
        self.set_pos(self.layout_circular())
        circular_tikz_latex = latex(self)
        opts.set_option('tkz_style', old_style)

        output = s.substitute(name=self._latex_escape(self.graph6_string()),
                              difficult=difficult_text,
                              order=self.order(),
                              size=self.size(),
                              alpha=self.independence_number(),
                              alphaproperties=alphaproperties_table,
                              lowerbounds=lowerbounds_table, 
                              upperbounds=upperbounds_table,
                              tikzpicture1=default_tikz_latex,
                              tikzpicture2=circular_tikz_latex)
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
            print "Creating PDF failed."

    @classmethod
    def _latex_escape(cls, str):
        # TODO: Write documentation
        # TODO: Write tests
        str = str.replace('\\', r'\textbackslash ')

        escape_chars = {
            '#': r'\#',
            '$': r'\$',
            '%': r'\%',
            '&': r'\&',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '^': r'\textasciicircum ',
            '~': r'\textasciitilde '
        }

        for old, new in escape_chars.iteritems():
            str = str.replace(old, new)

        str = str.replace('`', r'\`{}')    
        
        return str

    @classmethod
    def KillerGraph(cls):
        return cls('EXCO')

    @classmethod
    def ChairGraph(cls):
        return cls('DiC')

    @classmethod
    def CoChairGraph(cls):
        return cls('Dhw')

    @classmethod
    def PGraph(cls):
        return cls('Dl_')

    @classmethod
    def CoPGraph(cls):
        return cls('Dho')

    @classmethod
    def GemGraph(cls):
        return cls('Dh{')

    @classmethod
    def SuperClaw(cls, i, j, k):
        n = i + j + k + 1
        g = INPGraph(n)
        g.add_path([0] + range(n)[1:i+1])
        g.add_path([0] + range(n)[i+1:i+j+1])
        g.add_path([0] + range(n)[i+j+1:i+j+k+1])
        return g

    @classmethod
    def SkewStar(cls):
        return cls.SuperClaw(1,2,3)

    @classmethod
    def LatinSquareGraph(cls, ls, default_pos=False):
        r"""
        Construct a graph from the given Latin square.

        EXAMPLES:

        ::
            sage: from sage.combinat.matrices.latin import back_circulant
            sage: g = INPGraph.LatinSquareGraph(back_circulant(2))
            sage: g.is_isomorphic(graphs.CompleteGraph(4))
            True

        ::
            sage: g = INPGraph.LatinSquareGraph(matrix(ZZ, [[0,1],[1,0]]))
            sage: g.num_edges()
            6

        The Latin square can only contain symbols 0, ... n::
            sage: g = INPGraph.LatinSquareGraph([[1,2,3],[2,3,1],[3,1,2]])
            Traceback (most recent call last):
              ...
            ValueError: Not a Latin square.

        ::
            sage: g = INPGraph.LatinSquareGraph([[0,1,2],[1,2,0],[2,0,1]])
            sage: g.num_edges()
            27
        """
        if not isinstance(ls, LatinSquare):
            ls = LatinSquare(matrix(ZZ, ls))

        if not ls.is_latin_square():
            raise ValueError, "Not a Latin square."

        if ls.nrows() != ls.ncols():
            raise ValueError, "Matrix is not square."

        n = ls.nrows()
        g = INPGraph()
        pos = {}

        # Readjust the vertices on a bell curve to show off the row
        # and column cliques.
        f = lambda a, b, x: a * exp(-(b * (x - 0.5*(n-1)))**2)

        for i in range(n):
            for j in range(n):
                g.add_vertex((i, j, ls[i,j]))

                pos[(i, j, ls[i,j])] = [j+f(1.2,0.5,n-i-1), n-i-1+f(1.2,0.5,j)]
                for (row, col, val) in g.vertices():
                    if row == i and col == j: next
                    if row == i or col == j or val == ls[i,j]:
                        g.add_edge((row, col, val), (i, j, ls[i,j]))

        if not default_pos:
            g.set_pos(pos)

        return g

    @memoize_graphs
    def matching_number(self):
        r"""
        Compute the traditional matching number `\mu`, that is, the size of a
        maximum matching.

        EXAMPLES:

        ::
            sage: INPGraph(2).matching_number()
            0
            sage: INPGraph(graphs.CompleteGraph(3)).matching_number()
            1
            sage: INPGraph(graphs.PathGraph(3)).matching_number()
            1
            sage: INPGraph(graphs.StarGraph(3)).matching_number()
            1
            sage: INPGraph.KillerGraph().matching_number()
            2
            sage: INPGraph(graphs.CycleGraph(5)).matching_number()
            2
            sage: INPGraph(graphs.PetersenGraph()).matching_number()
            5

        WARNINGS:

        Ideally we would set use_edge_labels=False to ignore edge weighting and
        always compute the traditional matching number, but there is a bug
        in Sage 5.2 that returns double this number. Calling this on an
        edge-weighted graph will NOT give the usual matching number.
        """
        return int(self.matching(value_only=True, use_edge_labels=False))

    mu = matching_number

    def independence_number(self):
        r"""
        Compute the independence number using the Sage built-in independent_set
        method. This uses the Cliquer algorithm, which does not run in
        polynomial time.

        EXAMPLES:

        ::
            sage: INPGraph(2).independence_number()
            2
            sage: INPGraph(graphs.CompleteGraph(3)).independence_number()
            1
            sage: INPGraph(graphs.PathGraph(3)).independence_number()
            2
            sage: INPGraph(graphs.StarGraph(3)).independence_number()
            3

        You can also use :meth:alpha instead::
            sage: INPGraph.KillerGraph().alpha()
            4
            sage: INPGraph(graphs.CycleGraph(5)).alpha()
            2
            sage: INPGraph(graphs.PetersenGraph()).alpha()
            4
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
            sage: b = INPGraph(graphs.CompleteGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(graphs.CycleGraph(6))
            True
            sage: b = INPGraph(graphs.PathGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('EgCG'))
            True
            sage: b = INPGraph(graphs.StarGraph(3)).bipartite_double_cover()
            sage: b.is_isomorphic(Graph('Gs?GGG'))
            True
            sage: b = INPGraph('EXCO').bipartite_double_cover()
            sage: b.is_isomorphic(Graph('KXCO?C@??A_@'))
            True
            sage: b = INPGraph(graphs.CycleGraph(5)).bipartite_double_cover()
            sage: b.is_isomorphic(graphs.CycleGraph(10))
            True
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
        # TODO: Write documentation
        if isinstance(verts, list):
            neighborhood = []
            for v in verts:
                neighborhood += [v] + self.neighbors(v)
            return list(set(neighborhood))
        else:
            return [verts] + self.neighbors(verts)

    def closed_neighborhood_subgraph(self, verts):
        # TODO: Write tests
        # TODO: Write documentation
        return self.subgraph(self.closed_neighborhood(verts))

    def open_neighborhood(self, verts):
        # TODO: Write tests
        # TODO: Write documentation
        if isinstance(verts, list):
            neighborhood = []
            for v in verts:
                neighborhood += self.neighbors(v)
            return list(set(neighborhood))
        else:
            return self.neighbors(verts)

    def open_neighborhood_subgraph(self, verts):
        # TODO: Write tests
        # TODO: Write documentation
        return self.subgraph(self.open_neighborhood(verts))

    def max_degree(self):
        # TODO: Write tests
        # TODO: Write documentation
        return max(self.degree())

    def min_degree(self):
        # TODO: Write tests
        # TODO: Write documentation
        return min(self.degree())

    def is_stable_block(self, s):
        return self.is_independent_set(s) and \
            len(s) == self.closed_neighborhood_subgraph(s).independence_number()

    def stable_blocks(self, trivial=True):
        r"""
        Find all the stable blocks within the graph. A stable block is a set of
        vertices `S \in V(G)` such that `\alpha(G[S]) + \alpha(G[S^\text{c}]) = \alpha(G)`.

        EXAMPLES:

        Each vertex is its own stable block in a complete graph. ::
            sage: INPGraph(graphs.CompleteGraph(4)).stable_blocks()
            [[], [0], [1], [2], [3]]

        All subsets are stable in an empty graph. ::
            sage: INPGraph(3).stable_blocks()
            [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

        ::
            sage: INPGraph(graphs.CycleGraph(5)).stable_blocks()
            [[], [0, 2], [0, 3], [1, 3], [1, 4], [2, 4]]

        NOTES:
        This algorithm does not run in polynomial time.
        """
        # blocks = []
        # alpha = self.independence_number()
        # for k in range(1, alpha + 1):
        #     for S in combinations_iterator(self.vertices(), k):
        #         if self.is_independent_set(S):
        #             X = self.closed_neighborhood_subgraph(S)
        #             if X.independence_number() == k:
        #                 blocks.append(S)
        # return blocks

        if trivial:
            stability = lambda s: len(s) == self.closed_neighborhood_subgraph(s).independence_number()
        else:
            alpha = self.independence_number()
            stability = lambda s: len(s) > 0 and len(s) < alpha and len(s) == self.closed_neighborhood_subgraph(s).independence_number()

        return filter(stability, self.independent_sets())

    def independent_sets(self):
        r"""
        Return a list of all independent sets in the graph.

        NOTES:
        This is a naive algorithm and does not run in polynomial time.
        """
        alpha = self.independence_number()
        sets = [[]]
        for k in range(1, alpha + 1):
            # combinations_iterator is deprecated.
            #for S in combinations_iterator(self.vertices(), k):
            for S in Combinations(self.vertices(), k):
                if self.is_independent_set(S):
                    sets.append(S)
        return sets

    def critical_independent_sets(self):
        r"""
        Return a list of all critical independent sets in the graph.

        NOTES:
        This is a naive algorithm and does not run in polynomial time.
        """
        cis = {}
        for I in self.independent_sets():
            key = len(I) - len(self.open_neighborhood(I))
            if key in cis:
                cis[key].append(I)
            else:
                cis[key] = [I]
        return cis[max(cis.keys())]
    cis = critical_independent_sets

    def critical_independence_number(self):
        return max(len(I) for I in self.critical_independent_sets())
    alpha_c = critical_independence_number

    def block_survey(self):
        SB = self.stable_blocks()
        CIS = self.critical_independent_sets()
        IS = self.independent_sets()
        alpha = self.independence_number()
        alpha_c = self.critical_independence_number()

        for I in IS:
            output = str(I)
            if I in SB:
                output += " Stable"
            if I in CIS:
                output += " CIS"
                if len(I) == alpha_c:
                    output += " MaxCIS"
            if len(I) == alpha:
                output += " Max"
            print output

    def union_MCIS(self):
        r"""
        Return a union of maximum critical independent sets (MCIS).

        EXAMPLES:

        ::
            sage: INPGraph('Cx').union_MCIS()
            [0, 1, 3]
            sage: INPGraph(graphs.CycleGraph(4)).union_MCIS()
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

    def has_foldable_vertex(self):
        r"""
        Returns true if the graph has a foldable vertex, defined in
        Fomin-Grandoni-Kratsch 2006. A vertex `v` is foldable if `N(v)` contains
        no anti-triangles.

        EXAMPLES:

        ::
            sage: INPGraph('EqW_').has_foldable_vertex()
            True
            sage: INPGraph('G{O`?_').has_foldable_vertex()
            True

        For each vertex `v` in `K_{3,3}`, `N(v)` contains an anti-triangle ::
            sage: INPGraph(graphs.CompleteBipartiteGraph(3, 3)).has_foldable_vertex()
            False
        """
        # for v in self.vertices():
        #     if self.has_foldable_vertex_at(v):
        #         return True
        # return False

        # This any/imap version should be faster.
        return any(imap(self.has_foldable_vertex_at, self.vertices()))


    def has_foldable_vertex_at(self, v):
        r"""
        Returns true if `v` is a foldable vertex in the graph, defined in
        Fomin-Grandoni-Kratsch 2006. A vertex `v` is foldable if `N(v)` contains
        no anti-triangles.

        EXAMPLES:

        ::
            sage: INPGraph('EqW_').has_foldable_vertex_at(0)
            True
            sage: INPGraph('EqW_').has_foldable_vertex_at(1)
            False

        For each vertex `v` in `K_{3,3}`, `N(v)` contains an anti-triangle ::
            sage: INPGraph(graphs.CompleteBipartiteGraph(3, 3)).has_foldable_vertex_at(0)
            False
        """
        # Returns True if N(v) contains no anti-triangles
        return self.open_neighborhood_subgraph(v).complement().is_triangle_free()

    def fold_at(self, v):
        r"""
        Return a copy of the graph folded at `v`, as folding is defined in
        Fomin-Grandoni-Kratsch 2006.

        EXAMPLES:

        ::
            sage: G = INPGraph('EqW_')
            sage: G.fold_at(0).is_isomorphic(graphs.ClawGraph())
            True
            sage: G = INPGraph('G{O`?_')
            sage: G.fold_at(0).graph6_string()
            'E?dw'
        """
        if not self.has_foldable_vertex_at(v):
            raise ValueError, "The graph is not foldable at vertex " + str(v)

        g = self.copy()
        Nv = self.closed_neighborhood_subgraph(v)
        Nv_c = Nv.complement()
        new_nodes = []

        for (i,j) in Nv_c.edge_iterator(labels=False):
            g.add_vertex((i,j))
            g.add_edges([(i,j), w] for w in self.open_neighborhood([i, j]))
            g.add_edges([(i,j), w] for w in new_nodes)
            new_nodes += [(i,j)]

        g.delete_vertices(Nv.vertices())
        return g

    def is_bull_free(self):
        r"""
        Returns true if the graph is bull-free, that is, it does not contain
        any induced copies of the bull (a triangle with two pendants).

        EXAMPLES:

        ::
            sage: INPGraph(graphs.BullGraph()).is_bull_free()
            False
            sage: INPGraph('DyK').is_bull_free()
            True
            sage: INPGraph('EyGW').is_bull_free()
            False
        """
        return self.subgraph_search(graphs.BullGraph(), induced=True) is None

    def is_chair_free(self):
        r"""
        Returns true if the graph is chair-free, that is, it does not contain
        any induced copies of the chair.

        EXAMPLES:

        ::
            sage: INPGraph.ChairGraph().is_chair_free()
            False
            sage: INPGraph('Dic').is_chair_free()
            True
            sage: INPGraph('EiEG').is_chair_free()
            False
        """
        return self.subgraph_search(INPGraph.ChairGraph(), induced=True) is None

    def is_co_chair_free(self):
        return self.subgraph_search(INPGraph.CoChairGraph(), induced=True) is None

    def is_p5_free(self):
        return self.subgraph_search(graphs.PathGraph(5), induced=True) is None
    is_co_house_free = is_p5_free

    def is_house_free(self):
        return self.subgraph_search(graphs.HouseGraph(), induced=True) is None
    is_co_p5_free = is_house_free

    def is_p_free(self):
        return self.subgraph_search(INPGraph.PGraph(), induced=True) is None

    def is_co_p_free(self):
        return self.subgraph_search(INPGraph.CoPGraph(), induced=True) is None

    def is_gem_free(self):
        return self.subgraph_search(INPGraph.GemGraph(), induced=True) is None

    def is_p4_free(self):
        return self.subgraph_search(graphs.PathGraph(4), induced=True) is None
    is_co_gem_free = is_p4_free

    def is_diamond_free(self):
        return self.subgraph_search(graphs.DiamondGraph(), induced=True) is None

    def is_skew_star_free(self):
        return self.subgraph_search(INPGraph.SkewStar(), induced=True) is None

    ###########################################################################
    # Alpha properties
    ###########################################################################

    def has_max_degree_order_minus_one(self):
        r"""
        Returns true if the graph has a vertex with degree `n-1`, where `n` is
        the order of the graph.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).has_max_degree_order_minus_one()
            True
            sage: INPGraph(graphs.PathGraph(5)).has_max_degree_order_minus_one()
            False
        """
        return self.max_degree() == self.order() - 1
    has_max_degree_order_minus_one._is_alpha_property = True

    def is_claw_free(self):
        r"""
        Returns true is the graph is claw-free, that is, it does not contain
        any induced copies of the complete bipartite graph `K_{1,3}`.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).is_claw_free()
            True
            sage: INPGraph(graphs.StarGraph(5)).is_claw_free()
            False
            sage: INPGraph(graphs.ClawGraph()).is_claw_free()
            False
        """
        #return self.subgraph_search_count(graphs.ClawGraph()) == 0
        return self.subgraph_search(graphs.ClawGraph(), induced=True) is None
    is_claw_free._is_alpha_property = True

    def has_pendant_vertex(self):
        r"""
        Returns True if the graph contains a pendant vertex, that is, a vertex
        with degree 1.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).has_pendant_vertex()
            False
            sage: INPGraph(graphs.PathGraph(3)).has_pendant_vertex()
            True
        """
        return 1 in self.degree()
    has_pendant_vertex._is_alpha_property = True

    def has_simplicial_vertex(self):
        r"""
        Returns True if the graph has a simplicial vertex, that is, a vertex
        whose closed neighborhood forms a clique.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CycleGraph(4)).has_simplicial_vertex()
            False
            sage: INPGraph(graphs.CompleteGraph(4)).has_simplicial_vertex()
            True
        """
        # for v in self.vertices():
        #     if self.open_neighborhood_subgraph(v).is_clique():
        #         return True
        # return False

        # This any/imap version should be faster.
        neighborhood_is_clique = lambda v: self.open_neighborhood_subgraph(v).is_clique()
        return any(imap(neighborhood_is_clique, self.vertices()))
    has_simplicial_vertex._is_alpha_property = True

    @memoize_graphs
    def is_KE(self):
        r"""
        Determine if the graph is Konig-Egervary, that is, if `\alpha + \mu = n`.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.PathGraph(3)).is_KE()
            True
            sage: INPGraph(graphs.CycleGraph(3)).is_KE()
            False

        Not true that `\alpha_f + \mu = n` implies KE ::
            sage: INPGraph('GCpvdw').is_KE()
            False

        The graph `H_1` from Levit-Mandrescu 2011 is KE ::
            sage: INPGraph('Cx').is_KE()
            True

        The graph `H_2` from Levit-Mandrescu 2011 is also KE ::
            sage: INPGraph('FhcGO').is_KE()
            True

        But `H_3` from Levit-Mandrescu 2011 is not KE ::
            sage: INPGraph('DxC').is_KE()
            False
        """
        if self.is_bipartite():
            return True

        # c = self.union_MCIS()
        # # nc = []
        # # for v in c:
        # #     nc.extend(self.neighbors(v))
        # nc = self.open_neighborhood(c)

        # return list(set(c + nc)) == self.vertices()
        return self.vertices() == self.closed_neighborhood(self.union_MCIS())
    is_KE._is_alpha_property = True

    def is_almost_KE(self):
        # TODO: Write tests
        # TODO: Write documentation
        r"""
        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).is_almost_KE()
            True
            sage: INPGraph(graphs.CompleteGraph(4)).is_almost_KE()
            False

        This graph is almost KE, but not KE::
            sage: INPGraph('H?bF`xw').is_almost_KE()
            True

        """
        # subsets = combinations_iterator(self.vertices(), self.order() - 1)
        # for subset in subsets:
        #     if self.subgraph(subset).is_KE():
        #         return True

        # return False

        # This any/imap version should be faster.
        subgraph_is_KE = lambda s: self.subgraph(s).is_KE()
        
        # combinations_iterator is deprecated.
        # return any(imap(subgraph_is_KE, combinations_iterator(self.vertices(), self.order() - 1)))

        return any(imap(subgraph_is_KE, Combinations(self.vertices(), self.order() - 1)))
    is_almost_KE._is_alpha_property = True

    def has_nonempty_KE_part(self):
        # TODO: Write tests
        # TODO: Write documentation
        # if self.union_MCIS():
        #     return True
        # else:
        #     return False
        #return bool(self.union_MCIS())

        # We don't need to create the whole union of MCIS, we can stop if
        # one vertex satisfies it.
        # TODO: Can we speed this up further by removing copying?
        b = self.bipartite_double_cover()
        alpha = b.order() - b.matching_number()

        for v in self.vertices():
            test = b.copy()
            test.delete_vertices(b.closed_neighborhood([(v,0), (v,1)]))
            alpha_test = test.order() - test.matching_number() + 2
            if alpha_test == alpha:
                return True

        return False
    has_nonempty_KE_part._is_alpha_property = True

    def is_fold_reducible(self):
        # TODO: Write tests
        # TODO: Write documentation
        if not self.has_foldable_vertex():
            return False

        for v in self.vertices():
            if self.has_foldable_vertex_at(v):
                #if self.fold_at(v).order() < n:
                # We should be able to estimate this without actually folding
                Nv = self.closed_neighborhood_subgraph(v)
                Nv_c = Nv.complement()
                if Nv_c.size() - Nv.order() < 0:
                    return True
        return False
    is_fold_reducible._is_alpha_property = True

    def has_magnet(self):
        r"""
        Return true if the graph contains a magnetically-attracted pair, that is,
        adjacent vertices `a` and `b` such that `N(a) \setminus N(b)` is completely
        linked to `N(b) \setminus N(a)`. This definition is stated in
        Leveque-de Werra 2011.

        EXAMPLES:

        The Petesen Graph does not contain magnets ::
            sage: INPGraph(graphs.PetersenGraph()).has_magnet()
            False

        The killer does ::
            sage: INPGraph.KillerGraph().has_magnet()
            True
        """
        for a, b in self.edge_iterator(labels=False):
            Na_minus_Nb = set(self.neighbors(a)).difference(self.neighbors(b))
            Nb_minus_Na = set(self.neighbors(b)).difference(self.neighbors(a))

            # Check if completely linked
            if all(self.has_edge(u,v) for u in Na_minus_Nb for v in Nb_minus_Na):
                return True

        return False
    has_magnet._is_alpha_property = True

    def is_forbidden_subgraph_free(self):
        results = {
            'co_gem_free':      self.is_co_gem_free(),
            'claw_free':        self.is_claw_free(),
            'chair_free':       self.is_chair_free(),
            'p5_free':          self.is_p5_free(),
            'p_free':           self.is_p_free(),
            'co_p_free':        self.is_co_p_free(),
            'bull_free':        self.is_bull_free(),
            'co_chair_free':    self.is_co_chair_free(),
            'house_free':       self.is_house_free(),
            'gem_free':         self.is_gem_free(),
            'diamond_free':     self.is_diamond_free(),
            'skew_star_free':   self.is_skew_star_free()
        }

        return any([
                results['co_gem_free'],
                results['claw_free'],
                results['chair_free'],
                results['skew_star_free'],
                (results['p5_free'] and results['co_gem_free']),
                (results['p5_free'] and results['chair_free']),
                (results['p5_free'] and results['co_p_free']),
                (results['p5_free'] and results['p_free']),
                (results['p5_free'] and results['bull_free']),
                (results['p5_free'] and results['co_chair_free']),
                (results['p5_free'] and results['house_free']),
                (results['p5_free'] and results['gem_free']),
                (results['p5_free'] and results['diamond_free'])
            ])

    is_forbidden_subgraph_free._is_alpha_property = True

    ###########################################################################
    # Lower bounds
    ###########################################################################

    def matching_lower_bound(self):
        # TODO: Write more tests
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
        # TODO: Write documentation
        seq = self.degree_sequence()

        while seq[0] > 0:
            d = seq.pop(0)
            seq[:d] = [k-1 for k in seq[:d]]
            seq.sort(reverse=True)

        return len(seq)
    residue._is_lower_bound = True

    def average_degree_bound(self):
        # TODO: Write tests
        # TODO: Write documentation
        n = Integer(self.order())
        d = Rational(self.average_degree())
        return n / (1 + d)
    average_degree_bound._is_lower_bound = True

    def caro_wei(self):
        r"""
        Return the Caro-Wei lower bound.

        EXAMPLES:

        ::
            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.caro_wei()
            1
            sage: G = INPGraph(graphs.PathGraph(3))
            sage: G.caro_wei()
            4/3
        """
        return sum(1/(1+Integer(d)) for d in self.degree())
    caro_wei._is_lower_bound = True

    def seklow(self):
        # TODO: Write tests
        r"""
        Return Seklow's lower bound, which is an improvement on the Caro-Wei
        bound.

        """
        coeff = lambda v: Integer(1)/(1 + self.degree(v))
        return sum(coeff(v) * (1 + max(0, self.degree(v) * coeff(v) - \
            sum(coeff(w) for w in self.neighbors(v)))) for v in self.vertices())
    seklow._is_lower_bound = True

    def wilf(self):
        # TODO: Write tests
        # TODO: Write documentation
        n = Integer(self.order())
        max_eigenvalue = max(self.spectrum())
        if max_eigenvalue not in QQ:
            max_eigenvalue = RR(max_eigenvalue)
        return n / (1 + max_eigenvalue)
    wilf._is_lower_bound = True

    def hansen_zheng_lower_bound(self):
        # TODO: Write tests
        # TODO: Write documentation
        n = Integer(self.order())
        e = Integer(self.size())
        return ceil(n - (2 * e)/(1 + floor(2 * e / n)))
    hansen_zheng_lower_bound._is_lower_bound = True

    def harant(self):
        # TODO: Write tests
        # TODO: Write documentation
        n = Integer(self.order())
        e = Integer(self.size())
        term = 2 * e + n + 1
        return 0.5 * (term - sqrt(term**2 - 4*n**2))
    harant._is_lower_bound = True

    def max_even_minus_even_horizontal(self):
        r"""
        Compute `max\{e(v) - eh(v)}`, where `e(v)` is the number of vertices
        at even distance from vertex `v`, and `eh(v)` is the number of even
        horizontal edges with respect to `v`, that is, the number of edges `e`
        where both endpoints of `e` are at even distance from `v`.

        EXAMPLES:

        This isn't defined for disconnected graphs::
            sage: INPGraph(2).max_even_minus_even_horizontal()
            Traceback (most recent call last):
              ...
            ValueError: This bound is not defined for disconnected graphs.

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).max_even_minus_even_horizontal()
            1
            sage: INPGraph(graphs.PathGraph(3)).max_even_minus_even_horizontal()
            2
            sage: INPGraph.KillerGraph().max_even_minus_even_horizontal()
            3
            sage: INPGraph(graphs.CycleGraph(5)).max_even_minus_even_horizontal()
            2
        """
        if not self.is_connected():
            raise ValueError, "This bound is not defined for disconnected graphs."

        dist = self.distance_all_pairs()
        even = lambda v: [w for w in self.vertices() if dist[v][w] % 2 == 0]
        eh = lambda v: self.subgraph(even(v)).size()

        return max(len(even(v)) - eh(v) for v in self.vertices())
    max_even_minus_even_horizontal._is_lower_bound = True

    def max_odd_minus_odd_horizontal(self):
        r"""
        Compute `max\{o(v) - oh(v)}`, where `o(v)` is the number of vertices
        at odd distance from vertex `v`, and `oh(v)` is the number of odd
        horizontal edges with respect to `v`, that is, the number of edges `e`
        where both endpoints of `e` are at odd distance from `v`.

        EXAMPLES:

        This isn't defined for disconnected graphs::
            sage: INPGraph(2).max_odd_minus_odd_horizontal()
            Traceback (most recent call last):
              ...
            ValueError: This bound is not defined for disconnected graphs.

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).max_odd_minus_odd_horizontal()
            1
            sage: INPGraph(graphs.PathGraph(3)).max_odd_minus_odd_horizontal()
            2
            sage: INPGraph.KillerGraph().max_odd_minus_odd_horizontal()
            3
            sage: INPGraph(graphs.CycleGraph(5)).max_odd_minus_odd_horizontal()
            2
        """
        if not self.is_connected():
            raise ValueError, "This bound is not defined for disconnected graphs."

        dist = self.distance_all_pairs()
        odd = lambda v: [w for w in self.vertices() if dist[v][w] % 2 == 1]
        oh = lambda v: self.subgraph(odd(v)).size()

        return max(len(odd(v)) - oh(v) for v in self.vertices())
    max_odd_minus_odd_horizontal._is_lower_bound = True    

    def five_fourteenths_lower_bound(self):
        # TODO: Write documentation
        # TODO: Write tests
        if not (self.is_triangle_free() and self.max_degree() <= 3):
            raise ValueError, "This bound is only defined for triangle-free graphs of maximum degree at most 3."

        return 5 * self.order() / Integer(14)
    five_fourteenths_lower_bound._is_lower_bound = True

    def szekeres_wilf(self):
        pass


    def angel_campigotto_laforest(self):
        # TODO: Write tests
        r"""
        Compute the lower bound given in Angel-Campigotto-Laforest 2012.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).angel_campigotto_laforest()
            1
            sage: INPGraph(graphs.StarGraph(3)).angel_campigotto_laforest()
            8/3
        """
        n = self.order()
        c = len(self.connected_components())
        d = lambda u: Integer(self.degree(u))


        expected_size = n - sum(Integer(1)/(d(u) + 1) for u in self.vertices())

        if expected_size == (n - c):
            return c
        else:
            d_uv = lambda u, v: Integer(len(set(self.neighbors(u)).intersection(self.neighbors(v))))

            variance = sum(d(u)/((d(u) + 1)**2) for u in self.vertices()) - \
                       2 * sum(1/((d(u)+1)*(d(v)+1)) for u, v in self.edge_iterator(labels=False)) + \
                       2 * sum(d_uv(u,v)/((d(u)+1)*(d(v)+1)*(2+d(u)+d(v)-d_uv(u,v))) for u, v in self.complement().edge_iterator(labels=False))
            
            return n - (expected_size - variance/(n - c - expected_size))

    angel_campigotto_laforest._is_lower_bound = True

    ###########################################################################
    # Upper bounds
    ###########################################################################

    def matching_upper_bound(self):
        # TODO: Write more tests
        r"""
        Compute the matching number upper bound.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).matching_upper_bound()
            2
        """
        return self.order() - self.matching_number()
    matching_upper_bound._is_upper_bound = True

    def fractional_alpha(self):
        # TODO: Write more tests
        r"""
        Compute the fractional independence number of the graph.

        EXAMPLES:

        ::
            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.fractional_alpha()
            1.5
            sage: G = INPGraph(graphs.PathGraph(3))
            sage: G.fractional_alpha()
            2.0
        """
        p = MixedIntegerLinearProgram(maximization=True)
        x = p.new_variable()
        p.set_objective(sum(x[v] for v in self.vertices()))

        for v in self.vertices():
            p.add_constraint(x[v], max=1)

        for (u,v) in self.edge_iterator(labels=False):
            p.add_constraint(x[u] + x[v], max=1)

        return p.solve()
    fractional_alpha._is_upper_bound = True

    def lovasz_theta(self):
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

        # The below code assumes vertices are numbered 0, ..., n-1.
        gc = self.complement().relabel(inplace=False)
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

        # TODO: Rounding here is a total hack, sometimes it can come in slightly
        # under the analytical answer, for example, 2.999998 instead of 3, which
        # screws up the floor() call when checking difficult graphs.
        return round(v[0], 3)
    lovasz_theta._is_upper_bound = True

    def kwok(self):
        # TODO: Write more tests
        r"""
        Compute the upper bound `\alpha \leq n - \frac{e}{\Delta}` that is
        credited to Kwok, or possibly "folklore."

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).kwok()
            3/2
            sage: INPGraph(graphs.PathGraph(3)).kwok()
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
        # TODO: Write more tests
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
            sage: G = INPGraph(graphs.PathGraph(4))
            sage: G.min_degree_bound()
            3
        """
        return self.order() - self.min_degree()
    min_degree_bound._is_upper_bound = True

    def cvetkovic(self):
        # TODO: Write more tests
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
        eigenvalues = self.spectrum()
        positive = 0
        negative = 0
        zero = 0
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
        r"""
        Compute the annhilation number of the graph.

        EXAMPLES:

        ::
            sage: G = INPGraph(graphs.CompleteGraph(3))
            sage: G.annihilation_number()
            1
            sage: G = INPGraph(graphs.StarGraph(3))
            sage: G.annihilation_number()
            3
        """
        seq = sorted(self.degree())

        a = 0
        while sum(seq[:a+1]) <= sum(seq[a+1:]):
            a += 1

        return a
    annihilation_number._is_upper_bound = True

    def borg(self):
        # TODO: Write more tests
        r"""
        Compute the upper bound given by Borg.

        EXAMPLES:

        ::
            sage: INPGraph(graphs.CompleteGraph(3)).borg()
            2
        """
        n = Integer(self.order())
        Delta = Integer(self.max_degree())

        if Delta == 0:
            raise ValueError("Borg bound is not defined for graphs with maximum degree 0.")

        return n - ceil((n-1) / Delta)
    borg._is_upper_bound = True

    def cut_vertices_bound(self):
        # TODO: Write more tests
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

    _alpha_properties = [has_magnet, Graph.is_perfect, has_simplicial_vertex, is_forbidden_subgraph_free, has_nonempty_KE_part, is_almost_KE, is_fold_reducible]
    _lower_bounds = [angel_campigotto_laforest, Graph.radius, Graph.average_distance, five_fourteenths_lower_bound, max_even_minus_even_horizontal, max_odd_minus_odd_horizontal, matching_lower_bound, residue, average_degree_bound, caro_wei, seklow, wilf, hansen_zheng_lower_bound, harant]
    _upper_bounds = [matching_upper_bound, fractional_alpha, lovasz_theta, kwok, hansen_zheng_upper_bound, min_degree_bound, cvetkovic, annihilation_number, borg, cut_vertices_bound]
