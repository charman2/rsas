.. rsas documentation master file, created by
   sphinx-quickstart on Thu Jan 15 15:32:47 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation of the rsas library
=================================

The rsas theory is described in:

Harman, C. J. (2014), Time-variable transit time distributions and transport:
Theory and application to storage-dependent transport of chloride in a watershed,
Water Resour. Res., 51, doi:10.1002/2014WR015707.

The code in _rsas.pyx contains three slightly different implementations of the model.

rsas.solve_all_by_age_2out
    This is the original implementation used to generate the results in the paper.
    It solves for two outputs (Q1 and Q2, which might be discharge and ET) using
    an algorithm with an outer loop over all ages, and vectorized calculations over
    all times. It is slightly faster than the other implementations, but is more
    memory intensive. Unlike the others though, there is no option to calculate 
    output concentration timeseries inline. The calculated transit time
    distributions must be used to perform the convolutions after the code has 
    completed.
   
rsas.solve_all_by_time_2out
    Solution is found by looping over times, with all age calculations
    vectorized. Slower, but easier to understand and build on than 
    solve_all_by_age_2out. Includes option to determine output concentrations
    from a given input concentration progressively.
   
rsas.solve_all_by_time_1out
    Same as solve_all_by_time_2out, but for only one flux out (Q1). 
    
The end of the code gives class definitions for different rSAS functional forms.
These can be expanded with more definitions. The function rSAS_setup selects 
the requested functional form (using an identifier string) and initializes an
instance.

Documentation for the Code
**************************

.. automodule:: _rsas
   :members:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

