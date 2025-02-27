from synthetic_graph.graph_cython import SumDoubleInt, Input

a = Input([4])
b = Input([2], 1)
c = SumDoubleInt()
a >> c
b >> c
c.to_c_struct()
print(c.run())