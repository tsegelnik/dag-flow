import time
import numpy as np
import timeit
# from synthetic_graph.graph_python.library import Integrator, Sin, Sum, Product, Input, Cosh, Tan, Sinh, Repeater
from synthetic_graph.graph_cython.main import Integrator, Sin, Sum, Product, Input, Cosh, Tan, Sinh, Repeater

np.random.seed(33)

print("CREATE INPUT NODES")
NUM_NODES = 10000
ARRAY_SIZE = 5000

input_nodes = []
for i in range(NUM_NODES):
    data = np.random.rand(ARRAY_SIZE)
    input_nodes.append(Input(data))

print("CREATE GRAPH")
MULTIPLY_NODES_OPERATIONS = [Sum, Product, Integrator]
ONE_NODE_OPERATIONS = [Sin, Cosh,  Tan, Sinh]

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
        if operation == Integrator:
            input_nodes[current_nodes] >> node
            input_nodes[current_nodes + 1] >> node
            node = Repeater(ARRAY_SIZE, [node])
        else:
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

print("RUN GRAPH")
TESTS_COUNT = 10

start_time = time.time()
print("Average time: {:.4f}".format(
    timeit.timeit(input_nodes[0], number=TESTS_COUNT) / TESTS_COUNT
))
