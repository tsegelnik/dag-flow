#!/usr/bin/env python

from dagflow.storage import NodeStorage
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.graphviz import savegraph

from numpy import arange, meshgrid
from pytest import mark

@mark.parametrize('dtype', ('d', 'f'))
def test_to_root(testname, debug_graph, dtype):
    sizex = 12
    sizey = 10
    data = arange(sizex, dtype=dtype)+100
    data2 = arange(sizex*sizey, dtype=dtype).reshape(sizex, sizey)+100

    nodesx = arange(sizex, dtype=dtype)+0.5
    edgesx = arange(sizex+1, dtype=dtype)

    nodesy = arange(sizey, dtype=dtype)+0.5
    edgesy = arange(sizey+1, dtype=dtype)

    meshx, meshy = meshgrid(nodesx, nodesy, indexing='ij')

    with Graph(close=True, debug=debug_graph) as graph, NodeStorage({}) as storage:
        EdgesX = Array('edgesx', edgesx)
        EdgesY = Array('edgesy', edgesy)

        NodesX = Array('nodesx', nodesx)

        MeshX = Array('meshx', meshx)
        MeshY = Array('meshy', meshy)

        arr1e = Array('array edges', data, edges=EdgesX.outputs[0])
        arr1n = Array('array mesh', data)
        arr1n.outputs[0].dd.axes_meshes = (NodesX.outputs[0],)
        arr1n.outputs[0].dd.meshes_inherited = False

        arr2e = Array('array 2d edges', data2, edges=(EdgesX.outputs[0], EdgesY.outputs[0]))
        arr2n = Array('array 2d mesh', data2)
        arr2n.outputs[0].dd.axes_meshes = (MeshX.outputs[0], MeshY.outputs[0])
        arr2n.outputs[0].dd.meshes_inherited = False

    savegraph(graph, f"output/{testname}.pdf")

