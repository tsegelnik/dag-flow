import time
import numpy as np
import timeit
# from synthetic_graph.graph_python.library import Sin, Sum, Product, Input, Cosh, Tan, Sinh, Repeater
# from synthetic_graph.graph_ctypes.library import Sin, Sum, Product, Input, Cosh
from library import Sin, Sum, Product, Input, Cosh

np.random.seed(33)

print("CREATE INPUT NODES")
NUM_NODES = 10000
ARRAY_SIZE = 5000

input_nodes = []
for i in range(NUM_NODES):
    data = np.random.rand(ARRAY_SIZE)
    input_nodes.append(Input(data))

print("CREATE GRAPH")
MULTIPLY_NODES_OPERATIONS = [Sum, Product]
ONE_NODE_OPERATIONS = [Sin, Cosh]

while len(input_nodes) > 1:
    max_nodes = len(input_nodes)

    current_nodes = 0
    output_nodes = []
    while current_nodes < max_nodes:
        # 95% chance that there will be a node accepting only one input to make the graph deeper
        if np.random.rand() < 0.95:
            operation = np.random.choice(ONE_NODE_OPERATIONS)
        else:
            operation = np.random.choice(MULTIPLY_NODES_OPERATIONS)

        nodes_count = 2
        node = operation()

        if max_nodes - current_nodes <= 6:
            nodes_count = max_nodes - current_nodes
        elif operation in ONE_NODE_OPERATIONS:
            nodes_count = 1
        else:
            nodes_count = np.random.randint(1, 6)

        for i in range(nodes_count):
            input_nodes[current_nodes + i] >> node

        output_nodes.append(node)
        current_nodes += nodes_count

    input_nodes = output_nodes

node = input_nodes[0].compile()

print("RUN GRAPH")
TESTS_COUNT = 10

print("Average time: {:.4f}".format(
    timeit.timeit(input_nodes[0].run, number=TESTS_COUNT) / TESTS_COUNT
))

# print("Average time: {:.4f}".format(
#     timeit.timeit(input_nodes[0], number=TESTS_COUNT) / TESTS_COUNT
# ))


