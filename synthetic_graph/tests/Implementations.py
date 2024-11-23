from enum import Enum

class Implementations(Enum):
    CYTHON = "cython"
    CTYPES = "ctypes"
    PYTHON = "python"
    PYTHON_CTYPES = "python-ctypes"