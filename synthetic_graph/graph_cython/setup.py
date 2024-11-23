from setuptools import setup, Extension
from Cython.Build import cythonize
import os

source_dir = "src"
build_dir = os.path.join("build", "lib")
extensions = [
    Extension(
        "*",
        [os.path.join(source_dir, "*.pyx")],
    )
]

setup(
    ext_modules=cythonize(extensions, build_dir=build_dir),
    script_args=["build_ext", "--build-lib", build_dir],
)