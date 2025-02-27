cdef enum DataType:
    TYPE_INT = 1
    TYPE_FLOAT = 2
    TYPE_DOUBLE = 3


cdef extern from *:
    """
    #define SWITCH_TYPE(data_type, code_block) \\
        switch(data_type) { \\
            case DataType.TYPE_INT: { \\
                typedef int curr_type; \\
                code_block; \\
                break; \\
            } \\
            case DataType.TYPE_FLOAT: { \\
                typedef float curr_type; \\
                code_block; \\
                break; \\
            } \\
            case DataType.TYPE_DOUBLE: { \\
                typedef double curr_type; \\
                code_block; \\
                break; \\
            } \\
        }
    """
    pass
