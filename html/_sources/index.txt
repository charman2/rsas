.. rsas documentation master file, created by
   sphinx-quickstart on Thu Jan 15 15:32:47 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   TO UPDATE THE DOCUMENTATION:
	- Go into sourcetree, open the rsas repo
	- make sure you are in the right branch, and everything is committed
	- make sure the checked-out version is installed
	- open a terminal at rsas/docs and do $ make html
	- Go into sourcetree, open the gh-pages repo
	- Commit and push the changes
	

Documentation of the rsas library
=================================

This library allows you to model transport through arbitrary control volumes using
rank StorAge Selection (rSAS) functions.


Getting started
===============

Before you install
******************

rsas depends on the Python libraries numpy, scipy, and cython, and the example
codes use pandas to wrangle the timeseries data. These must all be installed.
The Anaconda package (https://store.continuum.io/cshop/anaconda/) contains all
the needed pieces, and is an easy way to get started. Install it before you
install rsas.

Getting rsas
************

rsas is available from github: https://github.com/charman2/rsas

The repository is currently private, so only collaborators can access it. Please
do not share the source code before I release it.

The code can be downloaded zipped, but a git clone is recommended so updates
can be provided as they are developed.

Installation
************

The main part of the code is written in Cython to allow fast execution. Before
you use rsas you must comile and install it. Open a terminal in the rsas directory
and run:

> python setup.py install

It may take a few minutes. You may get warning messages, all of which can be ignored.
Error messages cannot though, and will prevent the compilation from completion.

Once the code has compiled successfully you don't need to do it again
unless this code is changed.

Examples
********

Example uses of rsas are available in the ./examples directory. These should 
run right out of the box.

steady.py
-------------------------------

.. literalinclude:: ../examples/steady.py

steady2.py
------------------------

.. literalinclude:: ../examples/steady2.py

unsteady.py
------------------------

.. literalinclude:: ../examples/unsteady.py

Further reading
===============

The rSAS theory is described in:

Harman, C. J. (2014), Time-variable transit time distributions and transport:
Theory and application to storage-dependent transport of chloride in a watershed,
Water Resour. Res., 51, doi:10.1002/2014WR015707.


Documentation for the code
**************************

.. automodule:: rsas
   :members: solve, create_function, transport, transport_with_evapoconcentration

.. toctree::
   :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

