from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from multikeydict.nestedmkdict import NestedMKDict

from ..core.labels import repr_pretty
from ..tools.logger import logger
from . import input_handler
from .exception import ConnectionError
from .input import Input, Inputs
from .iter import StopNesting
from .output import Output, Outputs
from .shift import rshift

if TYPE_CHECKING:
    from .node import Node


class NodeBase:
    __slots__ = ("inputs", "outputs", "__missing_input_handler")
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
                sethandler = getattr(input_handler, handler)(self)
            elif isinstance(handler, type):
                sethandler = handler(self)
            else:
                sethandler = handler
                sethandler.node = self
        elif hasattr(self, "missing_input_handler"):
            sethandler = self.missing_input_handler
        else:
            sethandler = input_handler.MissingInputFail(self)
        self.__missing_input_handler = sethandler

    def __getitem__(self, key):
        if isinstance(key, (int, slice, str)):
            return self.outputs[key]
        if (y := len(key)) != 2:
            raise ValueError(f"NodeBase key should be of length 2, but given {y}!")
        ikey, okey = key
        if ikey and okey:
            if isinstance(ikey, (int, str)):
                ikey = (ikey,)
            if isinstance(okey, (int, str)):
                okey = (okey,)
            return NodeBase(
                self.inputs[ikey],
                self.outputs[okey],
                missing_input_handler=self.__missing_input_handler,
            )
        if ikey:
            return self.inputs[ikey]
        if okey:
            return self.outputs[okey]
        raise ValueError("Empty keys specified")

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except Exception:
            return default

    def __str__(self) -> str:
        return f"→[{len(self.inputs)}]{type(self).__name__}[{len(self.outputs)}]→"

    _repr_pretty_ = repr_pretty

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

    def __rshift_sequence(self, other: Sequence[Input]):
        # TODO: should choose only one possible option
        if len(self.outputs) == 1:
            # raise ConnectionError("NodeBase>>Tuple only supported when NodeBase has only 1 positional output", node=self)
            output = self.outputs[0]
            for input in other:
                output >> input
        elif len(self.outputs) == len(other):
            for output, input in zip(self.outputs, other):
                output >> input
        else:
            raise ConnectionError(
                f"Inconsistent outputs/inputs: {len(self.outputs), len(other)}",
                node=self,
            )

    def __rshift__(
        self,
        other: Input | Node | NodeBase | Sequence[Input] | Mapping[str, Output] | NestedMKDict,
    ):
        """
        self >> other
        """
        from .node import Node

        if isinstance(other, Input):
            if len(self.outputs) != 1:
                raise ConnectionError(
                    "NodeBase>>Input only supported when NodeBase has only 1 positional output",
                    node=self,
                )
            self.outputs[0] >> other
        elif isinstance(other, Sequence):
            self.__rshift_sequence(other)
        elif isinstance(other, (Mapping, NestedMKDict)):
            raise RuntimeError("Outdated logick. Reimplement!")
            # for name, output in self.outputs.iter_kw_items():
            #     try:
            #         input = other[name]
            #     except KeyError as e:
            #         raise ConnectionError(f"Unable to find input {name}", node=self) from e
            #     else:
            #         output >> input
        elif isinstance(other, NodeBase):
            return rshift(self, other)
        else:
            raise ConnectionError(f"Unsupported >>RHS type: {type(other)}", node=self)

    def __rrshift__(self, other: Output | Sequence[Output]):
        """
        other >> self
        """
        if not isinstance(other, Output):
            return rshift(other, self)
        for input in self.inputs:
            other >> input

    def __lshift__(self, storage: Mapping[str, Output]) -> None:
        """
        self << other

        For each not connected input try to find output with the same name in storage, then connect.
        """
        for name, inputs in self.inputs.all_edges.items():
            output = storage.get(name, None)
            if output is None:
                continue
            elif not isinstance(output, Output):
                output = getattr(output, "output", None)  # TODO: ugly, try something else
                if not isinstance(output, Output):
                    raise ConnectionError('[<<] invalid "output"', input=inputs, output=output)

            if not isinstance(inputs, Sequence):
                inputs = (inputs,)

            for input in inputs:
                if not input.connected():
                    logger.debug(f"[<<] connect {name}")
                    output >> input

    #
    # Accessors
    #
    def get_data(self, key=0):
        return self.outputs[key].data

    def get_input_data(self, key):
        return self.inputs[key].data()

    def to_dict(self, **kwargs) -> dict:
        return self.outputs[0].to_dict(**kwargs)
