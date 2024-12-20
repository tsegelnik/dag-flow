# import  tracemalloc
# tracemalloc.start()

import numpy as np
import time

from synthetic_graph.graph_cython import Sum, Input, Node

np.random.seed(33)

def make_test_graph(datasize=1, width=6, length=7):
    nsums = 0
    prevlayer = []

    data = np.random.uniform(-100, 100, size=datasize)
    data_node = Input(data)

    for ilayer in reversed(range(length)):
        ilayer_next = ilayer - 1
        n_groups = int(width ** ilayer_next)
        thislayer = []

        for igroup in range(n_groups):
            head = Sum()
            nsums += 1

            if prevlayer:
                for array in prevlayer:
                    array >> head
            else:
                for isource in range(width):
                    data_node >> head

            thislayer.append(head)

        prevlayer = thislayer

    return nsums, data_node, head

# tracemalloc.reset_peak()
nsums, data_node, head = make_test_graph(datasize=1, width=7, length=7)
print(nsums)
# create_size, create_peak = tracemalloc.get_traced_memory()
# print(create_size, create_peak)

# tracemalloc.reset_peak()
start_time = time.time()
head.to_c_struct()
print(time.time() - start_time)
# allocate_size, allocate_peak = tracemalloc.get_traced_memory()
# print(allocate_size, allocate_peak)