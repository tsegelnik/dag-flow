"""Test missing input handlers"""

from pytest import mark, raises

from dagflow.core.graph import Graph
from dagflow.core.input_strategy import (
    AddNewInput,
    AddNewInputAddAndKeepSingleOutput,
    AddNewInputAddNewOutput,
    AddNewInputAddNewOutputForBlock,
    InputStrategyBase,
)
from dagflow.lib.common import Dummy
from dagflow.plot.graphviz import savegraph

# TODO: add a test for AddNewInputAddNewOutputForNInputs


@mark.parametrize("strategy", (None, InputStrategyBase))
def test_InputStrategyBase(strategy, testname):
    """Test default strategy: fail on connect"""
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

            s = Dummy("add", input_strategy=strategy)
    graph.close()

    with raises(RuntimeError):
        nodes[0].outputs >> s
    with raises(RuntimeError):
        (node.outputs for node in nodes[:-1]) >> s
    savegraph(graph, f"output/{testname}.pdf", label="Fail on connect")


def test_AddNewInput(testname):
    """Test AddNewInput strategy: add new input on each new connect"""
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInput(output_kws={"allocatable": False}),
        )

    (node.outputs for node in nodes[:-1]) >> s
    nodes[-1].outputs >> s

    print()
    print(testname)
    s.print()
    graph.close()

    savegraph(graph, f"output/{testname}.pdf", label="Add only inputs")


def test_AddNewInputAddNewOutput(testname):
    """
    Test AddNewInputAddNewOutput strategy: add new input on each new connect
    and connect them as inputs to another input
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddNewOutput(output_kws={"allocatable": False}),
        )

    (node.outputs for node in nodes[:-1]) >> s
    nodes[-1].outputs >> s

    print()
    print(testname)
    s.print()

    for input, output in zip(s.inputs, s.outputs):
        assert input.child_output is output
    graph.close()

    savegraph(
        graph,
        f"output/{testname}.pdf",
        label="Add inputs and an output for each input",
    )


@mark.parametrize("child_output", (False, True))
def test_AddNewInputAddAndKeepSingleOutput(testname, child_output):
    """
    Test AddNewInputAddAndKeepSingleOutput strategy: add new input on each new connect and
    add an output if needed
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddAndKeepSingleOutput(
                add_child_output=child_output, output_kws={"allocatable": False}
            ),
        )

    (node.outputs for node in nodes[:-1]) >> s
    nodes[-1].outputs >> s

    print()
    print(testname)
    s.print()

    assert len(s.outputs) == 1
    if child_output:
        output = s.outputs[0]
        for input in s.inputs:
            assert input.child_output is output
    graph.close()

    savegraph(
        graph,
        f"output/{testname}.pdf",
        label="Add only inputs and only one output",
    )


@mark.parametrize("child_output", (False, True))
def test_AddNewInputAddNewOutputForBlock(testname, child_output):
    """
    Test AddNewInputAddNewOutputForBlock strategy: add new input on each new connect and
    add an output for each >> group
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddNewOutputForBlock(
                add_child_output=child_output, output_kws={"allocatable": False}
            ),
        )

    #(node.outputs for node in nodes[:-1]) >> s
    #nodes[-1].outputs >> s
    (node.outputs[0] for node in nodes[:-1]) >> s
    print(s.input_strategy._scope)
    nodes[-1].outputs[0] >> s
    print(s.input_strategy._scope)

    print()
    print(testname)
    s.print()

    assert len(s.outputs) == 2
    if child_output:
        o1, o2 = s.outputs
        for input in s.inputs[:3]:
            assert input.child_output is o1
        for input in s.inputs[3:]:
            assert input.child_output is o2
    graph.close()

    savegraph(
        graph,
        f"output/{testname}.pdf",
        label="Add inputs and an output for each block",
    )
