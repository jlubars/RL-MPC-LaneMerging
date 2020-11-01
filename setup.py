from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("st_cy.pyx", compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()]
)