from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("measles_efficiency.py")
)