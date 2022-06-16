from __future__ import print_function

import input_extra
from .shift import rshift, lshift
from .tools import StopNesting


class Legs:
    __missing_input_handler = None

    def __init__(self, inputs=None, outputs=None, missing_input_handler=None):
        self._missing_input_handler = missing_input_handler

        from .input import Inputs
        from .output import Outputs

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
        else:
            sethandler = input_extra.MissingInputFail(self)

        self.__missing_input_handler = sethandler

    def __getitem__(self, key):
        if isinstance(key, (int, slice, str)):
            return self.outputs[key]

        if len(key) != 2:
            raise ValueError("Legs key should be of length 2")

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

    def __str__(self):
        return f"->[{len(self.inputs)}],[{len(self.outputs)}]->"

    def _deep_iter_outputs(self):
        return iter(self.outputs)

    def _deep_iter_inputs(self, disconnected_only=False):
        return iter(self.inputs)

    def _deep_iter_corresponding_outputs(self):
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
