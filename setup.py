# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:57:18 2014

@author: ciaran
"""

try:
        from setuptools import setup
        from setuptools import Extension
except ImportError:
        from distutils.core import setup
        from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

config = {
    'description': 'Time-variable transport using storage selection (SAS) functions',
    'author': 'Ciaran J. Harman',
    'url': '',
    'download_url': '',
    'author_email': 'charamn1@jhu.edu',
    'version': '0.1.1',
    'install_requires': ['nose', 'numpy', 'scipy', 'cython'],
    'packages' : ['rsas'],
    'scripts': [],
    'name': 'rsas',
    'ext_modules': [Extension('_rsas_functions', ['./rsas/_rsas_functions.pyx'],
                              include_dirs=[numpy.get_include()],
                              libraries=None),
                    Extension('_rsas', ['./rsas/_rsas.pyx'],
                              include_dirs=[numpy.get_include()],
                              libraries=None)],
    'cmdclass' : { 'build_ext': build_ext }
}

setup(**config)

