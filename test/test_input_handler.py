#!/usr/bin/env python
"""Test missing input handlers"""

from contextlib import suppress

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import *
from dagflow.wrappers import *

nodeargs = dict(typefunc=lambda: True)


def test_00():
    """Test default handler: fail on connect"""
    graph = Graph()

    in1 = graph.add_node("n1", **nodeargs)
    in2 = graph.add_node("n2", **nodeargs)
    in3 = graph.add_node("n3", **nodeargs)
    in4 = graph.add_node("n4", **nodeargs)
    for node in (in1, in2, in3, in4):
        node.add_output("o1", allocatable=False)

    s = graph.add_node(
        "add", missing_input_handler=MissingInputFail, **nodeargs
    )
    graph.close()

    with suppress(Exception):
        (in1, in2, in3) >> s
    savegraph(
        graph, "output/missing_input_handler_00.pdf", label="Fail on connect"
    )


def test_01():
    """Test InputAdd handler: add new input on each new connect"""
    graph = Graph()

    in1 = graph.add_node("n1", **nodeargs)
    in2 = graph.add_node("n2", **nodeargs)
    in3 = graph.add_node("n3", **nodeargs)
    in4 = graph.add_node("n4", **nodeargs)
    for node in (in1, in2, in3, in4):
        node.add_output("o1", allocatable=False)

    s = graph.add_node(
        "add",
        missing_input_handler=MissingInputAdd(
            output_kws={"allocatable": False}
        ),
        **nodeargs
    )

    (in1, in2, in3) >> s
    in4 >> s

    print()
    print("test 01")
    s.print()
    graph.close()

    savegraph(
        graph, "output/missing_input_handler_01.pdf", label="Add only inputs"
    )


def test_02():
    """
    Test InputAddPair handler: add new input on each new connect
    and connect them as inputs to another input
    """
    graph = Graph()

    in1 = graph.add_node("n1", **nodeargs)
    in2 = graph.add_node("n2", **nodeargs)
    in3 = graph.add_node("n3", **nodeargs)
    in4 = graph.add_node("n4", **nodeargs)
    for node in (in1, in2, in3, in4):
        node.add_output("o1", allocatable=False)

    s = graph.add_node(
        "add",
        missing_input_handler=MissingInputAddPair(
            output_kws={"allocatable": False}
        ),
        **nodeargs
    )

    (in1, in2, in3) >> s
    in4 >> s

    print()
    print("test 02")
    s.print()

    for input, output in zip(s.inputs, s.outputs):
        assert input.child_output is output
    graph.close()

    savegraph(
        graph,
        "output/missing_input_handler_02.pdf",
        label="Add inputs and an output for each input",
    )


def test_03():
    """
    Test InputAddOne handler: add new input on each new connect and
    add an output if needed
    """
    graph = Graph()

    in1 = graph.add_node("n1", **nodeargs)
    in2 = graph.add_node("n2", **nodeargs)
    in3 = graph.add_node("n3", **nodeargs)
    in4 = graph.add_node("n4", **nodeargs)
    for node in (in1, in2, in3, in4):
        node.add_output("o1", allocatable=False)

    s = graph.add_node(
        "add",
        missing_input_handler=MissingInputAddOne(
            output_kws={"allocatable": False}
        ),
        **nodeargs
    )

    (in1, in2, in3) >> s
    in4 >> s

    print()
    print("test 03")
    s.print()
    graph.close()

    savegraph(
        graph,
        "output/missing_input_handler_03.pdf",
        label="Add only inputs and only one output",
    )


def test_04():
    """
    Test InputAddOne handler: add new input on each new connect and
    add an output if needed.
    This version also sets the input for each input
    """
    graph = Graph()

    in1 = graph.add_node("n1", **nodeargs)
    in2 = graph.add_node("n2", **nodeargs)
    in3 = graph.add_node("n3", **nodeargs)
    in4 = graph.add_node("n4", **nodeargs)
    for node in (in1, in2, in3, in4):
        node.add_output("o1", allocatable=False)

    s = graph.add_node(
        "add",
        missing_input_handler=MissingInputAddOne(
            add_child_output=True, output_kws={"allocatable": False}
        ),
        **nodeargs
    )

    (in1, in2, in3) >> s
    in4 >> s

    print()
    print("test 04")
    s.print()

    output = s.outputs[0]
    for input in s.inputs:
        assert input.child_output is output
    graph.close()

    savegraph(
        graph,
        "output/missing_input_handler_04.pdf",
        label="Add inputs and only one output",
    )


def test_05():
    """
    Test InputAddEach handler: add new input on each new connect and
    add an output for each >> group
    """
    graph = Graph()

    in1 = graph.add_node("n1", **nodeargs)
    in2 = graph.add_node("n2", **nodeargs)
    in3 = graph.add_node("n3", **nodeargs)
    in4 = graph.add_node("n4", **nodeargs)
    for node in (in1, in2, in3, in4):
        node.add_output("o1", allocatable=False)

    s = graph.add_node(
        "add",
        missing_input_handler=MissingInputAddEach(
            add_child_output=False, output_kws={"allocatable": False}
        ),
        **nodeargs
    )

    (in1, in2, in3) >> s
    in4 >> s

    print()
    print("test 05")
    s.print()
    graph.close()

    savegraph(
        graph,
        "output/missing_input_handler_05.pdf",
        label="Add inputs and an output for each block",
    )


def test_06():
    """
    Test InputAddEach handler: add new input on each new connect and
    add an output for each >> group.
    This version also sets the child_output for each input
    """
    graph = Graph()

    in1 = graph.add_node("n1", **nodeargs)
    in2 = graph.add_node("n2", **nodeargs)
    in3 = graph.add_node("n3", **nodeargs)
    in4 = graph.add_node("n4", **nodeargs)
    for node in (in1, in2, in3, in4):
        node.add_output("o1", allocatable=False)

    s = graph.add_node(
        "add",
        missing_input_handler=MissingInputAddEach(
            add_child_output=True, output_kws={"allocatable": False}
        ),
        **nodeargs
    )

    (in1, in2, in3) >> s
    in4 >> s

    print()
    print("test 06")
    s.print()

    o1, o2 = s.outputs
    for input in s.inputs[:3]:
        assert input.child_output is o1
    for input in s.inputs[3:]:
        assert input.child_output is o2
    graph.close()

    savegraph(
        graph,
        "output/missing_input_handler_06.pdf",
        label="Add inputs and an output for each block",
    )
