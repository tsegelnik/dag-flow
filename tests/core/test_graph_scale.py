from __future__ import annotations

from contextlib import suppress
from time import process_time
from typing import TYPE_CHECKING

from numpy import exp, log10
from numpy.random import uniform

from dagflow.graph import Graph
from dagflow.lib import Array, Sum

if TYPE_CHECKING:
    from dagflow.node import Node
    from numpy.typing import NDArray


def _make_data(datasize: int) -> NDArray:
    rnd_range = -100, 100
    return exp(uniform(*rnd_range, size=datasize))


def _make_test_graph(datasize: int = 1, width: int = 6, length: int = 7):
    Class = Sum
    ilayers = tuple(reversed(range(length)))

    nsums = 0
    input_arrays = []
    with Graph(close_on_exit=True) as g:
        prevlayer: list[Node] = []
        for ilayer in ilayers:
            ilayer_next = ilayer - 1
            n_groups = int(width**ilayer_next)

            thislayer: list[Node] = []

            for igroup in range(n_groups):
                head = Class(f"l={ilayer_next}, i={igroup}")
                nsums += 1
                # print(f"l={ilayer_next}, i={igroup}")
                thislayer.append(head)

                if prevlayer:
                    prevlayer = list(reversed(prevlayer))
                    for isource in range(width):
                        array = prevlayer.pop()
                        array >> head
                else:
                    for isource in range(width):
                        data = _make_data(datasize)
                        array = Array(f"l={ilayer}, g={igroup}, i={isource}", array=data)
                        input_arrays.append(array)
                        array >> head

                        # print(f"l={ilayer}, g={igroup}, i={isource}")

            prevlayer = thislayer

    return nsums, g, input_arrays, head


def _report(t1, t2, nsums, datasize):
    dt_ms = (t2 - t1) * 1000
    dt_rel_μs = dt_ms / nsums * 1e3
    print(f"Calculation time:")
    print(f"    time={dt_ms} ms")
    print(f"    time/sum={dt_rel_μs} μs")
    print(f"    {nsums} sums of {datasize} elements")

    return dt_ms, dt_rel_μs


def test_graph_scale_01(testname, width: int = 6, length: int = 7):
    datasize = 1
    nsums, g, input_arrays, head = _make_test_graph(datasize, width, length)

    size = []
    time_ms = []
    time_rel_μs = []
    for newdatasize in (None, 3, 10, 30, 100, 300, 1000, 3000, 10000):
        if newdatasize is not None:
            datasize = newdatasize
            g.open(open_nodes=True)
            for array in input_arrays:
                data = _make_data(datasize)
                array.outputs[0]._set_data(data, owns_buffer=True, override=True)
            g.close()

        assert head.tainted
        t1 = process_time()
        data = head.get_data()
        t2 = process_time()

        dt_ms, dt_rel_μs = _report(t1, t2, nsums, datasize)

        size.append(datasize)
        time_ms.append(dt_ms)
        time_rel_μs.append(dt_rel_μs)

    logsize = log10(size)
    print(size)
    print(logsize)
    print(time_ms)
    print(time_rel_μs)
    with suppress(Exception):
        import plotille

        fig = plotille.Figure()
        fig.x_label = "log₁₀(size)"
        fig.y_label = "time, ms"
        fig.width = 60
        fig.height = 30
        fig.set_x_limits(min_=0, max_=6)
        # fig.set_y_limits(min_=0, max_=1000)
        fig.color_mode = "byte"
        fig.plot(logsize, time_ms, lc=200)
        fig.scatter(logsize, time_ms, lc=200, marker="x")
        print(fig.show())

        fig = plotille.Figure()
        fig.x_label = "log₁₀(size)"
        fig.y_label = "time/N, ns"
        fig.width = 60
        fig.height = 30
        fig.set_x_limits(min_=0, max_=6)
        # fig.set_y_limits(min_=0, max_=1000)
        fig.color_mode = "byte"
        fig.plot(logsize, time_rel_μs, lc=200)
        fig.scatter(logsize, time_rel_μs, lc=200, marker="x")
        print(fig.show())

    print(f"Minimal time per sum: {min(time_rel_μs)} μs")

    # from dagflow.graphviz import GraphDot
    # d = GraphDot(g)
    # ofile = f"output/{testname}.dot"
    # d.savegraph(ofile)
    # print(f"Save graph: {ofile}")


def test_graph_scale_02(width: int = 2, length: int = 18):
    datasize = 1
    nsums, _, input_arrays, head = _make_test_graph(datasize, width, length)

    t1 = process_time()
    input_arrays[0].taint()
    t2 = process_time()
    _report(t1, t2, nsums, datasize)

    head.touch()

    t1 = process_time()
    input_arrays[1].taint()
    t2 = process_time()
    _report(t1, t2, nsums, datasize)

    head.touch()

    t1 = process_time()
    input_arrays[0].taint()
    input_arrays[1].taint()
    t2 = process_time()
    _report(t1, t2, nsums, datasize)
    head.touch()

    t1 = process_time()
    input_arrays[0].taint()
    input_arrays[-1].taint()
    t2 = process_time()
    _report(t1, t2, nsums, datasize)
    head.touch()

    t1 = process_time()
    for inp in input_arrays:
        inp.taint()
    t2 = process_time()
    _report(t1, t2, nsums, datasize)
    head.touch()
