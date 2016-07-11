# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:57:18 2014

@author: ciaran
"""

#try:
        #from setuptools import setup
        #from setuptools import Extension
#except ImportError:
        #from distutils.core import setup
        #from distutils.extension import Extension
from numpy.distutils.core import setup
from numpy.distutils.core import Extension
import numpy

config = {
    'description': 'Time-variable transport using storage selection (SAS) functions',
    'author': 'Ciaran J. Harman',
    'url': '',
    'download_url': '',
    'author_email': 'charman1@jhu.edu',
    'version': '0.6',
    'install_requires': ['numpy', 'scipy'],
    'packages' : ['rsas'],
    'scripts': [],
    'name': 'rsas',
    'ext_modules': [Extension(name='f_solve', sources=['./rsas/solve.f90'],
                              include_dirs=[numpy.get_include()],
                              extra_f90_compile_args = ["-fcheck=all"],
                              libraries=None),
        Extension(name='f_convolve', sources=['./rsas/convolve.f90'],
                              include_dirs=[numpy.get_include()],
                              extra_f90_compile_args = ["-fcheck=all"],
                              libraries=None)],
}

setup(**config)

