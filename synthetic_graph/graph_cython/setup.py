from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "functions.pyx",
        "node.pyx",
        "library.pyx",
    ], annotate=True, language_level="3"),
)
