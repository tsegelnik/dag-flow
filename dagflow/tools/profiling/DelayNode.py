from typing import TYPE_CHECKING
from time import sleep

from dagflow.node import Node
from dagflow.inputhandler import MissingInputAdd
from dagflow.exception import InitializationError
if TYPE_CHECKING:
    from dagflow.output import Output

class DelayNode(Node):
    """
    A node that sleeps for a certain time.
    Used for profiler debugging and testing.

    use `sleep_time` argument to set execution time in seconds
    """
    _out: "Output"
    _sleep_time: float | int
    __slots__ = ("_out", "_sleep_time")

    def __init__(self, *args, sleep_time: float = 0.01, **kwargs):
        if not isinstance(sleep_time, (int, float)):
            raise InitializationError("Invalid sleep_time type "
                                      "(use `float` or `int`)")
        self._sleep_time = sleep_time
        kwargs.setdefault("missing_input_handler", MissingInputAdd())
        super().__init__(*args, **kwargs)
        self._out = self._add_output("result")

    def _fcn(self):
        for inp in self.inputs.iter_all():
            inp.touch()
        sleep(self._sleep_time)

    def _typefunc(self):
        self._out.dd.dtype = int
        self._out.dd.shape = ()
