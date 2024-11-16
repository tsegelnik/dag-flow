from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Указываем путь к исходным файлам
source_dir = "src"

# Указываем путь для сборки
build_dir = os.path.join("build", "lib")

# Создаем список расширений
extensions = [
    Extension(
        "*",
        [os.path.join(source_dir, "*.pyx")],
    )
]

setup(
    ext_modules=cythonize(extensions, build_dir=build_dir, annotate=True),
    script_args=["build_ext", "--build-lib", build_dir],
)