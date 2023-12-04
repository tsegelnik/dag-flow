from time import sleep

from dagflow.nodes import FunctionNode
from dagflow.input_extra import MissingInputAdd

class SleepyNode(FunctionNode):
    """
    A node that sleeps for a certain time. 
    Used for profiling debugging and testing.

    use `sleep_time` argument to set execution time in seconds
    """

    def __init__(self, *args, sleep_time=0.01, **kwargs):
        self._sleep_time = sleep_time
        kwargs.setdefault("missing_input_handler", MissingInputAdd())
        super().__init__(*args, **kwargs)
        self._out = self._add_output("result")
        self._out.dd.dtype = int
        self._out.dd.shape = ()

    def _fcn(self):
        for inp in self.inputs.iter_all():
            inp.touch()
        sleep(self._sleep_time)

    def _typefunc(self) -> bool:
        t_sleep_time = type(self._sleep_time)
        if t_sleep_time != int and t_sleep_time != float:
            raise TypeError("Invalid sleep_time type "
                            "(Note: use float)")
        