
from . import input_extra
from .input import Inputs
from .output import Outputs
from .shift import lshift, rshift
from .iter import StopNesting

class Legs:
    inputs: Inputs
    outputs: Outputs
    def __init__(self, inputs=None, outputs=None, missing_input_handler=None):
        self._missing_input_handler = missing_input_handler
        self.inputs = Inputs(inputs)
        self.outputs = Outputs(outputs)

    @property
    def _missing_input_handler(self):
        return self.__missing_input_handler

    @_missing_input_handler.setter
    def _missing_input_handler(self, handler):
        if handler:
            if isinstance(handler, str):
                sethandler = getattr(input_extra, handler)(self)
            elif isinstance(handler, type):
                sethandler = handler(self)
            else:
                sethandler = handler
                sethandler.node = self
        elif hasattr(self, 'missing_input_handler'):
            sethandler = self.missing_input_handler
        else:
            sethandler = input_extra.MissingInputFail(self)
        self.__missing_input_handler = sethandler

    def __getitem__(self, key):
        if isinstance(key, (int, slice, str)):
            return self.outputs[key]
        if (y := len(key)) != 2:
            raise ValueError(f"Legs key should be of length 2, but given {y}!")
        ikey, okey = key
        if ikey and okey:
            if isinstance(ikey, (int, str)):
                ikey = (ikey,)
            if isinstance(okey, (int, str)):
                okey = (okey,)
            return Legs(
                self.inputs[ikey],
                self.outputs[okey],
                missing_input_handler=self.__missing_input_handler,
            )
        if ikey:
            return self.inputs[ikey]
        if okey:
            return self.outputs[okey]
        raise ValueError("Empty keys specified")

    def get(self, key, default = None):
        try:
            return self.__getitem__(key)
        except Exception:
            return default

    def __str__(self) -> str:
        return f"→[{len(self.inputs)}],[{len(self.outputs)}]→"

    def __repr__(self) -> str:
        return self.__str__()

    def deep_iter_outputs(self):
        return iter(self.outputs)

    def deep_iter_inputs(self, disconnected_only=False):
        return iter(self.inputs)

    def deep_iter_child_outputs(self):
        raise StopNesting(self)

    def print(self):
        for i, input in enumerate(self.inputs):
            print(i, input)
        for i, output in enumerate(self.outputs):
            print(i, output)

    def __rshift__(self, other):
        """
        self >> other
        """
        return rshift(self, other)

    def __rlshift__(self, other):
        """
        other << self
        """
        return rshift(self, other)

    def __lshift__(self, other):
        """
        self << other
        """
        return lshift(self, other)

    def __rrshift__(self, other):
        """
        other >> self
        """
        return lshift(self, other)
