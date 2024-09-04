# to see output of this file you need use -s flag:
#       pytest -s ./test/tools/profiling/test_memoryprofiler.py
from __future__ import annotations
import types
from collections import Counter
from typing import TYPE_CHECKING
from functools import reduce
from operator import mul

from dagflow.tools.profiling import MemoryProfiler
from dagflow.input import Input
from dagflow.lib import Array
from test_helpers import graph_0, graph_1


if TYPE_CHECKING:
    from dagflow.output import Output
    from dagflow.input import Input
    from numpy.typing import NDArray

def calc_numpy_size(data: NDArray):
    """Size of Numpy's `NDArray` in bytes"""
    length = reduce(mul, data.shape)
    return length * data.dtype.itemsize

def get_input_size(inp: Input):
    if inp.owns_buffer:
        return calc_numpy_size(inp.own_data)
    return 0

def get_output_size(out: Output):
    if out.owns_buffer or (out.has_data and out._allocating_input is None):
        return calc_numpy_size(out.data_unsafe)
    return 0

def edge_size(edge: Output | Input):
    """Return size of `edge` data in bytes"""
    if isinstance(edge, Input):
        return get_input_size(edge)
    return get_output_size(edge)


def test_basic_edges_g0():
    g, nodes = graph_0()
    a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes

    # A2 (1 output)
    out: Output = a2.outputs['array'] # 'array' - default name for Array output
    
    actual_sizes = MemoryProfiler.estimate_node(a2)
    o_expected = edge_size(out)
    assert o_expected != 0
    assert o_expected == actual_sizes[out]

    # S1 (3 inputs, 2 outputs)
    actual_sizes = MemoryProfiler.estimate_node(s1)

    for inp in s1.inputs.iter_all():
        i_expected = edge_size(inp)
        assert i_expected == actual_sizes[inp]

    for out in s1.outputs.iter_all():
        o_expected = edge_size(out)
        assert o_expected == actual_sizes[out]


def test_array_store_mods_g0():
    """Test profiling behavior with different array store modes"""
    g, nodes = graph_0()
    a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes

    # P1 (2 inputs, 2 outputs)
    #  parent output from 'A1' has Array node type with 'store_weak' mode
    sw_out: Output = a1.outputs['array']
    conn_input: Input = p1.inputs[0]
    
    assert conn_input.parent_output == sw_out
    assert sw_out._allocating_input is None

    assert edge_size(conn_input) == 0
    assert edge_size(sw_out) != edge_size(conn_input)

    actual_sizes = MemoryProfiler.estimate_node(a1)
    assert edge_size(sw_out) == actual_sizes[sw_out]
    
    actual_sizes = MemoryProfiler.estimate_node(p1)
    assert edge_size(conn_input) == actual_sizes[conn_input]

    # S0 (2 inputs, 1 output)
    #  parent output from 'A3' has Array node type with 'fill' mode 
    f_out: Output = a3.outputs['array']
    conn_input: Input = s0.inputs[1]

    assert conn_input.parent_output == f_out
def test_estimate_all_edges():
    g, nodes = graph_0()
    
    mp = MemoryProfiler(nodes)
    mp.estimate_target_nodes()
    
    assert hasattr(mp, "_estimations_table")




