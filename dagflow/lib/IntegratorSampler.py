from typing import Literal, Optional
from numpy import empty, errstate, floating, integer, issubdtype, linspace
from numpy.typing import NDArray

from ..exception import InitializationError, TypeFunctionError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_inputs_number,
)


class IntegratorSampler(FunctionNode):
    """
    The `IntegratorSampler` node create sample for integration.

    There are two `dim`s for corresponding integrator mode: `1d` and `2d`.
    Also there are several samplers for `1d` (`rect`, `trap`, `gl`) and only `gl`
    for `2d` integrator, where `rect` is rectangular, `trap` is trapezoidal
    and `gl` for Gauss-Legendre.

    There is optional argument `offset` for the `rect` sampler,
    taking the following values: `left`, `center`, or `right`.

    There is only one positional input: `edges`.
    There are two outputs: 0 - `sample`, 1 - `weights`
    """

    __buffer: NDArray

    def __init__(
        self,
        *args,
        dim: int,
        mode: Literal["rect", "trap", "gl"],
        offset: Optional[Literal["left", "center", "right"]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if dim not in {1, 2}:
            raise InitializationError(
                f"Argument `dim` must be 1 or 2, but given '{dim}'!",
                node=self,
            )
        if mode not in {"rect", "trap", "gl"}:
            raise InitializationError(
                f"Argument `mode` must be 'rect', 'trap', or 'gl', but given '{mode}'!",
                node=self,
            )
        if mode != "rect" and offset is not None:
            raise InitializationError(
                "Argument `offset` is used only within 'rect' mode!", node=self
            )
        if dim == 2 and mode in {"trap", "rect"}:
            raise InitializationError(
                "Only `gl` mode is allowed for dim=2!", node=self
            )
        self._dim = dim
        self._mode = mode
        self._offset = offset if offset is not None else "center"
        self._add_input("ordersX", positional=False)
        if self._dim == 2:
            self._add_input("ordersY", positional=False)
        self._add_output(("sample", "weights"))
        self._functions.update(
            {
                "rect": self._fcn_rect,
                "trap": self._fcn_trap,
                "gl1d": self._fcn_gl1d,
                "gl2d": self._fcn_gl2d,
            }
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def offset(self) -> Optional[str]:
        return self._offset

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an integration algorithm,
        determines dtype and shape for outputs
        """
        check_inputs_number(self, 1)
        check_has_inputs(self, "ordersX")
        input0 = self.inputs[0]
        if (ndim := len(input0.dd.shape)) != self.dim:
            raise TypeFunctionError(
                f"The Integrator works only with {self.dim} inputs, but one has {ndim=}!",
                node=self,
            )
        check_input_dimension(self, slice(None), self.dim)
        check_input_dimension(self, "ordersX", 1)
        ordersX = self.inputs["ordersX"]
        if not issubdtype(ordersX.dd.dtype, integer):
            raise TypeFunctionError(
                "The `ordersX` must be array of integers, but given '{ordersX.dd.dtype}'!",
                node=self,
                input=ordersX,
            )
        dtype = input0.dd.dtype
        if not issubdtype(dtype, floating):
            raise TypeFunctionError(
                "The Integrator works only within `float` or `double` "
                f"precision, but given '{dtype}'!",
                node=self,
            )
        if sum(ordersX.data) != input0.dd.shape[0]:
            raise TypeFunctionError(
                "ordersX must be consistent with inputs shape, "
                f"but given {ordersX.data=} and {input0.dd.shape=}!",
                node=self,
                input=ordersX,
            )
        if self.dim == 2:
            check_has_inputs(self, "ordersY")
            check_input_dimension(self, "ordersY", 1)
            ordersY = self.inputs["ordersY"]
            if not issubdtype(ordersY.dd.dtype, integer):
                raise TypeFunctionError(
                    "The `ordersY` must be array of integers, but given '{ordersY.dd.dtype}'!",
                    node=self,
                    input=ordersY,
                )
            if sum(ordersY.data) != input0.dd.shape[1]:
                raise TypeFunctionError(
                    "ordersY must be consistent with inputs shape, "
                    f"but given {ordersY.data=} and {input0.dd.shape=}!",
                    node=self,
                    input=ordersX,
                )
        self.fcn = self._functions[self.mode]
        for output in self.outputs:
            output.dd.dtype = dtype
            output.dd.shape = input0.dd.shape

    def _post_allocate(self):
        """Allocates the `buffer`, which elements are the follows:
        * 0: binwidths
        * 1: samplewidths
        * 2: axis_nodes
        * 3: low edges (only for `rect`)
        * 4: high edges (only for `rect`)
        """
        input0 = self.inputs[0].dd
        if self.mode == "rect":
            shape = [input0.shape[0] - 1] * 5
        elif self.mode == "trap":
            shape = [input0.shape[0] - 1] * 3
        else:
            #TODO: configure the buffer for Gauss-Legendre
            shape = []
        self.__buffer = empty(shape=shape, dtype=input0.dtype)

    def _fcn_rect(self, _, inputs, outputs):
        """The rectangular sampling"""
        ordersX = outputs["ordersX"].data  # type: NDArray
        binwidths = self.__buffer[0]
        samplewidths = self.__buffer[1]
        nodes = self.__buffer[2]
        low = self.__buffer[3]
        high = self.__buffer[4]
        edges = inputs[0].data
        sample = outputs[0].data
        weights = outputs[1].data

        binwidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = binwidths / ordersX
        if self.offset == "left":
            low[:] = edges[:-1]
            high[:] = edges[1:] - samplewidths
        elif self.offset == "center":
            low[:] = edges[:-1] + samplewidths * 0.5
            high[:] = edges[1:] - samplewidths * 0.5
        else:
            low[:] = edges[:-1] + samplewidths
            high[:] = edges[1:]
        nodes[:] = (edges[1:] + edges[:-1]) * 0.5

        offset = 0
        for i, n in enumerate(ordersX):
            if n > 1:
                sample[offset:n] = linspace(low[i], high[i], n)
                weights[offset:n] = samplewidths[i]
            else:
                sample[offset:n] = low[i]
                weights[offset:n] = binwidths[i]
            offset += n

        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return [outputs.iter_data()]

    def _fcn_trap(self, _, inputs, outputs):
        """The trapezoidal sampling"""
        ordersX = outputs["ordersX"].data  # type: NDArray
        binwidths = self.__buffer[0]
        samplewidths = self.__buffer[1]
        nodes = self.__buffer[2]
        edges = inputs[0].data
        sample = outputs[0].data
        weights = outputs[1].data

        binwidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = binwidths / (ordersX - 1.0)
        nodes[:] = (edges[1:] + edges[:-1]) * 0.5

        offset = 0
        for i, n in enumerate(ordersX):
            sample[offset:n] = linspace(edges[i], edges[i + 1], n)
            weights[offset] = samplewidths[i] * 0.5
            if n > 2:
                weights[offset + 1 : n - 2] = samplewidths[i]
            offset += n - 1
        weights[-1] = samplewidths[-1] * 0.5

        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return [outputs.iter_data()]

    def _fcn_gl1d(self, _, inputs, outputs):
        """The 1d Gauss-Legendre sampling"""
        # TODO: implement GL 1d
        if self.debug:
            return [outputs.iter_data()]

    def _fcn_gl2d(self, _, inputs, outputs):
        """The 2d Gauss-Legendre sampling"""
        # TODO: implement GL 2d
        if self.debug:
            return [outputs.iter_data()]
