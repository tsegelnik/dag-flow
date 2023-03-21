
from .printl import next_level, printl


def printer(fcn, node, inputs, outputs):
    printl(f"Evaluate {node.name}")
    with next_level():
        fcn(node, inputs, outputs)
    printl(f"... done with {node.name}")


def before_printer(fcn, node, inputs, outputs):
    printl(f"Evaluate {node.name}: {node.label()}")
    with next_level():
        fcn(node, inputs, outputs)


def after_printer(fcn, node, inputs, outputs):
    with next_level():
        fcn(node, inputs, outputs)
    printl(f"Evaluate {node.name}: {node.label()}")


def dataprinter(fcn, node, inputs, outputs):
    fcn(node, inputs, outputs)
    for i, output in enumerate(outputs):
        printl("{: 2d} {}: {!s}".format(i, output.name, output._data))


def toucher(fcn, node, inputs, outputs):
    for i, input in enumerate(inputs):
        printl("touch input {: 2d} {}.{}".format(i, node.name, input.name))
        with next_level():
            input.touch()
    fcn(node, inputs, outputs)
