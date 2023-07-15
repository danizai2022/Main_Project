from distutils.core import setup, Extension
from Cython.Build import cythonize
import os
import numpy

#python setupCythons.py build_ext --inplace


for fname in os.listdir('.'):
    name, extention = os.path.splitext(fname)
    if extention == '.pyx':
        setup(ext_modules = cythonize(fname), 
              include_dirs=[numpy.get_include()])


