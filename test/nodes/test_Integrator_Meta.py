from dagflow.graph import Graph
from dagflow.meta_node import MetaNode
from dagflow.lib.Array import Array
from dagflow.lib.Integrator import Integrator
from dagflow.lib.IntegratorSampler import IntegratorSampler
from dagflow.lib.trigonometry import Cos, Sin
from numpy import allclose, linspace, pi

from dagflow.graphviz import savegraph

def test_Integrator_trap_meta(debug_graph):
    metaint = MetaNode()

    with Graph(debug=debug_graph, close=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1))
        ordersX = Array("ordersX", [1000] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])

        sampler = IntegratorSampler("sampler", mode="trap")
        integrator = Integrator("integrator")

        cosf = Cos("cos")
        sinf = Sin("sin")
        ordersX >> sampler("ordersX")
        sampler.outputs["x"] >> cosf
        A >> sinf
        B >> sinf
        sampler.outputs["weights"] >> integrator("weights")
        cosf.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
    res = sinf.outputs[1].data - sinf.outputs[0].data
    assert allclose(integrator.outputs[0].data, res, atol=1e-2)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]

    savegraph(graph, f"output/test_Integrator_trap_meta.pdf", show='all')
