import numpy as np

from dagflow.core.node import Node
from dagflow.core.graph import Graph
from dagflow.lib.common import Array
from dagflow.lib.linalg import MatrixProductDVDt
from dagflow.lib.arithmetic import Product, Sum
from dagflow.plot.graphviz import GraphDot

def graph_0() -> tuple[Graph, list[Node]]:
    with Graph(close_on_exit=True) as graph:
        a0 = Array("A0", [8, 7, 13])
        a1 = Array("A1", [1, 2, 4], mode='store_weak')
        a2 = Array("A2", [12, 22, 121])
        a3 = Array("A3", [4, 3, 3], mode='fill')

        p0 = Product("P0")
        p1 = Product("P1")
        s0 = Sum("S0")

        (a0, a1) >> p0
        (a1, a2) >> p1
        (a2, a3) >> s0

        l_matrix = Array("L-Matrix", [[5, 2, 1], [1, 4, 12], [31, 7, 2]])
        s1 = Sum("S1 (diagonal of S-Matrix )")
        s2 = Sum("S2")

        (p0, p1, s0) >> s1
        (p1) >> s2

        mdvdt = MatrixProductDVDt("Matrix Product DVDt")
        s3 = Sum("S3")

        l_matrix >> mdvdt.inputs["left"]
        s1 >> mdvdt.inputs["square"]
        (s1, s2) >> s3
    s3['result'].data
    mdvdt['result'].data

    nodes = [a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt]
    return graph, nodes

def graph_1() -> tuple[Graph, list[Node]]:
    with Graph(close_on_exit=True) as graph:
        array_nodes = [Array(f"A{i}", np.arange(i, i+3, dtype='f'))
                       for i in range(5)]
        s1 = Sum("S1")
        array_nodes[:3] >> s1

        s2 = Sum("S2")
        (array_nodes[2: 4]) >> s2 # ("A2", "A3") >> s2

        p1 = Product("p1")
        (array_nodes[4], s1) >> p1

        p2 = Product("p2")
        (s2, p1) >> p2
    p2["result"].data

    nodes = [*array_nodes, s1, s2, p1, p2]
    return graph, nodes

def test_exec_graph_0():
    _, nodes = graph_0()
    s3, mdvdt = nodes[-3], nodes[-1]
    s3_data = s3.outputs['result'].data_unsafe
    mdvdt_data = mdvdt.outputs['result'].data_unsafe
    # check for all zeros
    assert np.any(s3_data), "graph_0, `s3` was not evaluated"
    assert np.any(mdvdt_data), "graph_0, `mdvdt` was not evaluated"

def test_exec_graph_1():
    _, nodes = graph_1()
    p2 = nodes[-1]
    p2_data = p2.outputs['result'].data_unsafe
    assert np.any(p2_data), "graph_1, `p2` was not evaluated"

def test_invoke_and_save():
    graphs = [graph_0, graph_1]
    for i, g in enumerate(graphs):
        graph_dot = GraphDot(g()[0])
        graph_dot.savegraph(f"output/test_profiling_graph_{i}.png")
