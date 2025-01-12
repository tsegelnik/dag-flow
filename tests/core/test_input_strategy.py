"""Test missing input handlers"""

from pytest import mark, raises

from dagflow.core.graph import Graph
from dagflow.core.input_strategy import (
    AddNewInput,
    AddNewInputAddAndKeepSingleOutput,
    AddNewInputAddNewOutput,
    AddNewInputAddNewOutputForBlock,
    AddNewInputAddNewOutputForNInputs,
    InputStrategyBase,
)
from dagflow.core.output import Output, Outputs
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


@mark.parametrize("n_blocks", (1, 2, 3, 4))
@mark.parametrize("child_output", (False, True))
@mark.parametrize("connection_type", (Outputs, Output))
def test_AddNewInputAddNewOutputForBlock(testname, child_output, n_blocks, connection_type):
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

    get_output = lambda obj: obj if connection_type is Outputs else obj[0]
    match n_blocks:
        case 1:
            (get_output(node.outputs) for node in nodes) >> s
        case 2:
            (get_output(node.outputs) for node in nodes[:2]) >> s
            (get_output(node.outputs) for node in nodes[2:]) >> s
        case 3:
            for node in nodes[:2]:
                get_output(node.outputs) >> s
            (get_output(node.outputs) for node in nodes[2:]) >> s
        case 4:
            for node in nodes:
                get_output(node.outputs) >> s
        case _:
            pass

    print()
    print(testname)
    s.print()

    assert len(s.outputs) == n_blocks
    if child_output:
        match n_blocks:
            case 1:
                for inp in s.inputs:
                    assert inp.child_output is s.outputs[0]
            case 2:
                for inp in s.inputs[:2]:
                    assert inp.child_output is s.outputs[0]
                for inp in s.inputs[2:]:
                    assert inp.child_output is s.outputs[1]
            case 3:
                assert s.inputs[0].child_output is s.outputs[0]
                assert s.inputs[1].child_output is s.outputs[1]
                for inp in s.inputs[2:]:
                    assert inp.child_output is s.outputs[2]
            case 4:
                for inp, out in zip(s.inputs, s.outputs):
                    assert inp.child_output is out
            case _:
                pass
    graph.close()

    savegraph(
        graph,
        f"output/{testname}.pdf",
        label="Add inputs and an output for each block",
    )


@mark.parametrize("n_inp", (1, 2, 4))
@mark.parametrize("child_output", (False, True))
@mark.parametrize("connection_type", (Outputs, Output))
def test_AddNewInputAddNewOutputForNInputs(testname, child_output, n_inp, connection_type):
    """
    Test AddNewInputAddNewOutputForNInputs strategy: add new input on each new connect and
    add an output for each n inputs in >> operator
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddNewOutputForNInputs(
                n=n_inp, add_child_output=child_output, output_kws={"allocatable": False}
            ),
        )

    get_output = lambda obj: obj if connection_type is Outputs else obj[0]
    (get_output(node.outputs) for node in nodes) >> s
    assert len(s.outputs) == int(4 / n_inp)

    print()
    print(testname)
    s.print()

    if child_output:
        match n_inp:
            case 1:
                for inp, out in zip(s.inputs, s.outputs):
                    assert inp.child_output is out
            case 2:
                # TODO: is it correct? maybe 1 -> 0 and 2 -> 1?
                assert s.inputs[0].child_output is s.outputs[0]
                assert s.inputs[2].child_output is s.outputs[1]
            case 4:
                assert s.inputs[0].child_output is s.outputs[-1]
    graph.close()

    savegraph(
        graph,
        f"output/{testname}.pdf",
        label="Add inputs and an output for each block",
    )
