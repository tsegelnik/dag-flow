import time
import numpy as np
import timeit
from dagflow.graph import Graph
from dagflow.storage import NodeStorage
from dagflow.lib.Array import Array
from dagflow.lib.trigonometry import Sin, Cos
from dagflow.lib.arithmetic import Sum, Product
from dagflow.bundles.file_reader import FileReader
from dagflow.graphviz import savegraph

print("CREATE INPUT NODES")
NUM_NODES = 1000
ARRAY_SIZE = 500

MULTIPLY_NODES_OPERATIONS = [Sum, Product]
ONE_NODE_OPERATIONS = [Sin, Cos]

ONE_NODE_OPERATIONS_NAMES = ["sin", "cos"]

storage = NodeStorage()


with (
    Graph(close_on_exit=True) as graph,
    storage,
    FileReader
):
    data = np.random.rand(ARRAY_SIZE)
    data_node, _ = Array.replicate(
        name=f"array",
        array=data
    )

    print("CREATE GRAPH")

    op_nodes = []
    for i in range(NUM_NODES):
        op_node, _ = ONE_NODE_OPERATIONS[i % 2].replicate(
            name=f"operation_{i}_{ONE_NODE_OPERATIONS_NAMES[i % 2]}"
        )
        data_node >> op_node
        op_nodes.append(op_node)

    sum_nodes = []
    product_nodes = []
    for i in range(NUM_NODES // 20):
        sum_node, _ = Sum.replicate(
            name=f"sum_{i}",
        )
        sum_nodes.append(sum_node)
        product_node, _ = Product.replicate(
            name=f"product_{i}",
        )
        product_nodes.append(product_node)

    for i, op_node in enumerate(op_nodes):
        op_node >> sum_nodes[i // 20]
        op_node >> product_nodes[i // 20]

    for i in range(NUM_NODES // 20):
        sum_nodes[i] >> product_nodes[i]

    sum_final_node, _ = Sum.replicate(
        name=f"sum_final",
    )

    for node in sum_nodes + product_nodes:
        node >> sum_final_node


savegraph(graph, "dagflow_example_1.dot", show=["type", "mark", "label", "path"],)

print("RUN GRAPH")
TESTS_COUNT = 1

def test(node_start, node_end):
    node_start.taint()
    node_end.outputs[0].data

print("Average time: {:.4f}".format(
    timeit.timeit(lambda : test(data_node, sum_final_node), number=TESTS_COUNT) / TESTS_COUNT
))

# print("Average time: {:.4f}".format(
#     timeit.timeit(input_nodes[0], number=TESTS_COUNT) / TESTS_COUNT
# ))
