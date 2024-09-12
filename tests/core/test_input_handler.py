"""Test missing input handlers"""
from contextlib import suppress

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.inputhandler import MissingInputAdd
from dagflow.inputhandler import MissingInputAddEach
from dagflow.inputhandler import MissingInputAddOne
from dagflow.inputhandler import MissingInputAddPair
from dagflow.inputhandler import MissingInputFail
from dagflow.lib.Dummy import Dummy

# TODO: add a test for MissingInputAddEachN


def test_00():
    """Test default handler: fail on connect"""
    with Graph() as graph:
        in1 = Dummy("n1")
        in2 = Dummy("n2")
        in3 = Dummy("n3")
        in4 = Dummy("n4")
        for node in (in1, in2, in3, in4):
            node.add_output("o1", allocatable=False)

            s = Dummy("add", missing_input_handler=MissingInputFail)
    graph.close()

    with suppress(Exception):
        (in1, in2, in3) >> s
    savegraph(graph, "output/missing_input_handler_00.pdf", label="Fail on connect")


def test_01():
    """Test InputAdd handler: add new input on each new connect"""
    with Graph() as graph:
        in1 = Dummy("n1")
        in2 = Dummy("n2")
        in3 = Dummy("n3")
        in4 = Dummy("n4")
        for node in (in1, in2, in3, in4):
            node.add_output("o1", allocatable=False)

            s = Dummy(
                "add",
                missing_input_handler=MissingInputAdd(output_kws={"allocatable": False}),
            )

    (in1, in2, in3) >> s
    in4 >> s

    print()
    print("test 01")
    s.print()
    graph.close()

    savegraph(graph, "output/missing_input_handler_01.pdf", label="Add only inputs")


def test_02():
    """
    Test InputAddPair handler: add new input on each new connect
    and connect them as inputs to another input
    """
    with Graph() as graph:
        in1 = Dummy("n1")
        in2 = Dummy("n2")
        in3 = Dummy("n3")
        in4 = Dummy("n4")
        for node in (in1, in2, in3, in4):
            node.add_output("o1", allocatable=False)

            s = Dummy(
                "add",
                missing_input_handler=MissingInputAddPair(output_kws={"allocatable": False}),
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
    with Graph() as graph:
        in1 = Dummy("n1")
        in2 = Dummy("n2")
        in3 = Dummy("n3")
        in4 = Dummy("n4")
        for node in (in1, in2, in3, in4):
            node.add_output("o1", allocatable=False)

            s = Dummy(
                "add",
                missing_input_handler=MissingInputAddOne(output_kws={"allocatable": False}),
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
    with Graph() as graph:
        in1 = Dummy("n1")
        in2 = Dummy("n2")
        in3 = Dummy("n3")
        in4 = Dummy("n4")
        for node in (in1, in2, in3, in4):
            node.add_output("o1", allocatable=False)

            s = Dummy(
                "add",
                missing_input_handler=MissingInputAddOne(
                    add_child_output=True, output_kws={"allocatable": False}
                ),
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
    with Graph() as graph:
        in1 = Dummy("n1")
        in2 = Dummy("n2")
        in3 = Dummy("n3")
        in4 = Dummy("n4")
        for node in (in1, in2, in3, in4):
            node.add_output("o1", allocatable=False)

            s = Dummy(
                "add",
                missing_input_handler=MissingInputAddEach(
                    add_child_output=False, output_kws={"allocatable": False}
                ),
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
    with Graph() as graph:
        in1 = Dummy("n1")
        in2 = Dummy("n2")
        in3 = Dummy("n3")
        in4 = Dummy("n4")
        for node in (in1, in2, in3, in4):
            node.add_output("o1", allocatable=False)

            s = Dummy(
                "add",
                missing_input_handler=MissingInputAddEach(
                    add_child_output=True, output_kws={"allocatable": False}
                ),
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
