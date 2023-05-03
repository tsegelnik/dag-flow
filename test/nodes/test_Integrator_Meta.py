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
        sampler.outputs["weights"] >> integrator("weights")

        metaint.add_node(sampler, kw_inputs=['ordersX'], kw_outputs=['x'])
        metaint.add_node(integrator, kw_inputs=[('ordersX', 'ordersX1')],
                         inputs_pos=True, outputs_pos=True, missing_inputs=True, also_missing_outputs=True)

        cosf = Cos("cos")
        sinf = Sin("sin")
        ordersX >> metaint.inputs["ordersX"]
        ordersX >> metaint.inputs["ordersX1"]

        metaint.outputs["x"] >> cosf
        metaint.outputs["x"] >> sinf

        cosf.outputs[0] >> metaint
        sinf.outputs[0] >> metaint

        sincheck = Sin("sin")
        coscheck = Cos("cos")
        A >> sincheck
        B >> sincheck
        A >> coscheck
        B >> coscheck
    res1 =   sincheck.outputs[1].data - sincheck.outputs[0].data
    res2 = - coscheck.outputs[1].data + coscheck.outputs[0].data
    assert allclose(integrator.outputs[0].data, res1, rtol=0, atol=1e-2)
    assert allclose(integrator.outputs[1].data, res2, rtol=0, atol=1e-2)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    assert integrator.outputs[1].dd.axes_edges == [edges["array"]]

    savegraph(graph, f"output/test_Integrator_trap_meta.pdf", show='all')
