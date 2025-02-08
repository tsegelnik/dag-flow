from simple_node import SimpleNode
 
# Тест с int
node_int = SimpleNode([1, 2, 3], 1)
print("Int before:", node_int.get_data())
node_int.multiply_by_scalar(2)
print("Int after:", node_int.get_data())
    
# Тест с float
node_float = SimpleNode([1.5, 2.5, 3.5], 2)
print("Float before:", node_float.get_data())
node_float.multiply_by_scalar(2.0)
print("Float after:", node_float.get_data())
    
# Тест с double
node_double = SimpleNode([1.1, 2.2, 3.3], 3)
print("Double before:", node_double.get_data())
node_double.multiply_by_scalar(2.0)
print("Double after:", node_double.get_data())