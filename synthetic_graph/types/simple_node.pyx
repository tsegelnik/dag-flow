import cython
from libc.stdlib cimport malloc, free

cdef class SimpleNode:
    cdef:
        CNode* node
        public int data_type  # public для простоты тестирования
        
    # Константы для типов
    DEF TYPE_INT = 1
    DEF TYPE_FLOAT = 2
    DEF TYPE_DOUBLE = 3
    
    def __init__(self, data, int data_type):
        self.data_type = data_type
        self.node = self._create_node(data)
    
    cdef CNode* _create_node(self, data):
        cdef CNode* node = <CNode*>malloc(sizeof(CNode))
        cdef int i
        cdef int size = len(data)
        
        # Выделяем память под данные нужного типа
        if self.data_type == TYPE_INT:
            node.data = <void*>malloc(sizeof(int) * size)
            for i in range(size):
                (<int*>node.data)[i] = data[i]
        elif self.data_type == TYPE_FLOAT:
            node.data = <void*>malloc(sizeof(float) * size)
            for i in range(size):
                (<float*>node.data)[i] = data[i]
        elif self.data_type == TYPE_DOUBLE:
            node.data = <void*>malloc(sizeof(double) * size)
            for i in range(size):
                (<double*>node.data)[i] = data[i]
                
        node.size = size
        node.data_type = self.data_type
        return node
    
    def get_data(self):
        """Получить данные обратно в Python"""
        cdef int i
        result = []
        
        if self.data_type == TYPE_INT:
            for i in range(self.node.size):
                result.append((<int*>self.node.data)[i])
        elif self.data_type == TYPE_FLOAT:
            for i in range(self.node.size):
                result.append((<float*>self.node.data)[i])
        elif self.data_type == TYPE_DOUBLE:
            for i in range(self.node.size):
                result.append((<double*>self.node.data)[i])
                
        return result
    
    def multiply_by_scalar(self, value):
        """Простая операция для проверки работы с разными типами"""
        cdef int i
        
        if self.data_type == TYPE_INT:
            for i in range(self.node.size):
                (<int*>self.node.data)[i] *= int(value)
        elif self.data_type == TYPE_FLOAT:
            for i in range(self.node.size):
                (<float*>self.node.data)[i] *= float(value)
        elif self.data_type == TYPE_DOUBLE:
            for i in range(self.node.size):
                (<double*>self.node.data)[i] *= float(value)
    
    def __dealloc__(self):
        if self.node != NULL:
            if self.node.data != NULL:
                free(self.node.data)
            free(self.node)