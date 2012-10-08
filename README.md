inp
===

This module creates an `INPGraph` class which provides lots of useful methods
for computing graph invariants and bounds for independence number. It also
allows us to test whether or not a graph is "difficult," according to current
theory as well as search for difficult graphs.

Installation
------------
This module requires at least version 5.2 of Sage.

To perform any of the graph searches, you'll need to install the `nauty` package
into Sage:

`sage -i nauty`

You may also install the `python-progressbar` module to display pretty progress bars
while searching:

1. Download [python-progressbar](http://code.google.com/p/python-progressbar/) and unpack it
2. Install it into Sage's version of python: `sage -python setup.py install`

Examples
--------

The `INPGraph()` constructor recognizes all of the same arguments that the usual
Sage `Graph()` does:

`G = INPGraph(5) # an empty graph on 5 vertices`

You may also pass other graphs into the constructor:

`G = INPGraph(graphs.CompleteGraph(5))`

You can check a graph's independence number using the inefficient built-in method:

`G.independence_number()` or `G.alpha()`

Compute the Lovasz theta value for the graph:

`G.lovasz_theta()`

Test if a graph is difficult:

`G.is_difficult()`

Check a particular invariant against all graphs of a given order:

`INPGraph.survey(INPGraph.residue, 8)`

Search for a difficult graph:

`G = INPGraph.next_difficult_graph() # this will also create a PDF with information about the graph`