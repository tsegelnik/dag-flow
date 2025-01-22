from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence

from ..core.labels import repr_pretty
from ..tools.logger import logger
from .exception import ConnectionError, InitializationError
from .input import Inputs
from .input_strategy import InputStrateges, InputStrategyBase
from .output import Output, Outputs


class NodeBase:
    __slots__ = ("inputs", "outputs", "_input_strategy")
    inputs: Inputs
    outputs: Outputs

    def __init__(self, inputs=None, outputs=None, input_strategy=None):
        self.input_strategy = input_strategy
        self.inputs = Inputs(inputs)
        self.outputs = Outputs(outputs)

    @property
    def input_strategy(self):
        return self._input_strategy

    @input_strategy.setter
    def input_strategy(self, input_strategy):
        if input_strategy is None:
            self._input_strategy = InputStrategyBase()
        elif isinstance(input_strategy, InputStrategyBase):
            self._input_strategy = input_strategy
            self._input_strategy.node = self
        elif issubclass(input_strategy, InputStrategyBase):
            self._input_strategy = input_strategy(node=self)
        else:
            raise InitializationError(
                f"Wrong {input_strategy=}! Must be in {InputStrateges}", node=self
            )

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
                input_strategy=self._input_strategy,
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

    def print(self):
        for i, input in enumerate(self.inputs):
            print(i, input)
        for i, output in enumerate(self.outputs):
            print(i, output)

    def __rshift__(self, other):
        """
        self >> other

        .. note:: now the connection for the node supports only one output
        """
        if self.outputs.len_all() == 1:
            out = tuple(self.outputs.all_edges.values())[0]
        else:
            raise ConnectionError(
                f"The connection of {type(self)} >> {type(other)} is not implemented!"
                " The connection `Node >> ...` supported only for nodes with only 1 output!",
                node=self,
            )
        out >> other

    def __rrshift__(self, other: Sequence | Generator):
        """other >> self"""
        if not isinstance(other, (Sequence, Generator)):
            raise ConnectionError(
                f"The connection {type(other)=} >> {type(self)=} is not implemented", node=self
            )
        scope = self._input_strategy._scope + 1
        for out in other:
            if isinstance(out, (Output, Outputs)):
                out.connect_to_node(self, scope=scope, reassign_scope=False)
            else:
                out >> self
        self._input_strategy._scope = scope

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
