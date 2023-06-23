from .printl import next_level, printl


def printer(fcn, node):
    printl(f"Evaluate {node.name}")
    with next_level():
        fcn()
    printl(f"... done with {node.name}")


def before_printer(fcn, node):
    printl(f"Evaluate {node.name}: {node.label()}")
    with next_level():
        fcn()


def after_printer(fcn, node):
    with next_level():
        fcn()
    printl(f"Evaluate {node.name}: {node.label()}")


def dataprinter(fcn, node):
    fcn()
    for i, output in enumerate(node.outputs):
        printl("{: 2d} {}: {!s}".format(i, output.name, output._data))


def toucher(fcn, node):
    for i, _input in enumerate(node.inputs):
        printl("touch input {: 2d} {}.{}".format(i, node.name, _input.name))
        with next_level():
            _input.touch()
    fcn()
