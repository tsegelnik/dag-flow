from typing import Literal, Optional

from numpy import empty, errstate, integer, issubdtype, linspace
from numpy.polynomial.legendre import leggauss
from numpy.typing import DTypeLike, NDArray

from ..exception import InitializationError, TypeFunctionError
from ..nodes import FunctionNode
from ..typefunctions import check_input_dimension, check_inputs_number
from ..types import InputT


class IntegratorSampler(FunctionNode):
    """
    The `IntegratorSampler` node creates a sample for the `Integrator` node.

    There are several samplers for `1d` (`rect`, `trap`, `gl`) and only `2d`
    for `2d` integrator, where `rect` is the rectangular, `trap` is the trapezoidal,
    `gl` is the 1d Gauss-Legendre, and `2d` is the 2d Gauss-Legendre.

    There is optional argument `offset` for the `rect` sampler,
    taking the following values: `left`, `center`, or `right`.

    There is no positional inputs. It is supposed that `orders` already have `edges`.
    There are two outputs: 0 - `sample`, 1 - `weights`
    """

    __buffer: NDArray

    def __init__(
        self,
        *args,
        mode: Literal["rect", "trap", "gl", "2d"],
        dtype: DTypeLike = "d",
        offset: Optional[Literal["left", "center", "right"]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if mode not in {"rect", "trap", "gl", "2d"}:
            raise InitializationError(
                f"Argument `mode` must be 'rect', 'trap', 'gl', or '2d', but given '{mode}'!",
                node=self,
            )
        if offset is not None and mode != "rect":
            raise InitializationError(
                "Argument `offset` is used only within 'rect' mode!", node=self
            )
        self._dtype = dtype
        self._mode = mode
        self._offset = offset if offset is not None else "center"
        self._add_input("ordersX", positional=False)
        if mode == "2d":
            self._add_input("ordersY", positional=False)
        self._add_output(("sample", "weights"))
        self._functions.update(
            {
                "rect": self._fcn_rect,
                "trap": self._fcn_trap,
                "gl": self._fcn_gl1d,
                "2d": self._fcn_gl2d,
            }
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def offset(self) -> Optional[str]:
        return self._offset

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an integration algorithm,
        determines dtype and shape for outputs
        """
        check_inputs_number(self, 0)
        ordersX = self.__check_orders("ordersX")
        shape = ordersX.dd.shape[0]
        if self.mode == "2d":
            ordersY = self.__check_orders("ordersY")
            shape = [2, max((shape, ordersY.dd.shape[0]))]
        shape = tuple(shape)
        self.fcn = self._functions[self.mode]
        for output in self.outputs:
            output.dd.dtype = self.dtype
            output.dd.shape = shape

    def __check_orders(self, name: str) -> InputT:
        """
        The method checks dimension (==1) of the input `name`, type (==`integer`),
        and returns the input
        """
        check_input_dimension(self, name, 1)
        result = self.inputs[name]
        if not issubdtype(result.dd.dtype, integer):
            raise TypeFunctionError(
                f"The `name` must be array of integers, but given '{result.dd.dtype}'!",
                node=self,
                input=result,
            )
        return result

    def _post_allocate(self) -> None:
        """Allocates the `buffer`, which elements are the follows:
        * 0: axis_nodes
        * 1: binwidths
        * 2: samplewidths
        * 3: low edges (only for `rect`)
        * 4: high edges (only for `rect`)
        """
        edgeshape = self.inputs["ordersX"].dd.edges.shape[0]
        if self.mode == "rect":
            shape = [edgeshape - 1] * 5
        elif self.mode == "trap":
            shape = [edgeshape - 1] * 3
        elif self.mode == "gl":
            shape = [edgeshape - 1]
        else:
            # TODO: implement 2d GL sampling
            shape = []
        self.__buffer = empty(shape=shape, dtype=self.dtype)

    def _fcn_rect(self, _, inputs, outputs) -> Optional[list]:
        """The rectangular sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.edges
        orders = ordersX.data
        sample = outputs[0].data
        weights = outputs[1].data
        nodes = self.__buffer[0]
        binwidths = self.__buffer[1]
        samplewidths = self.__buffer[2]
        low = self.__buffer[3]
        high = self.__buffer[4]

        binwidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = binwidths / orders
        if self.offset == "left":
            low[:] = edges[:-1]
            high[:] = edges[1:] - samplewidths
        elif self.offset == "center":
            low[:] = edges[:-1] + samplewidths * 0.5
            high[:] = edges[1:] - samplewidths * 0.5
        else:
            low[:] = edges[:-1] + samplewidths
            high[:] = edges[1:]

        offset = 0
        for i, n in enumerate(orders):
            if n > 1:
                sample[offset : offset + n] = linspace(low[i], high[i], n)
                weights[offset : offset + n] = samplewidths[i]
            else:
                sample[offset : offset + n] = low[i]
                weights[offset : offset + n] = binwidths[i]
            offset += n

        nodes[:] = (edges[1:] + edges[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return [outputs.iter_data()]

    def _fcn_trap(self, _, inputs, outputs) -> Optional[list]:
        """The trapezoidal sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.edges
        orders = ordersX.data
        sample = outputs[0].data
        weights = outputs[1].data
        nodes = self.__buffer[0]
        binwidths = self.__buffer[1]
        samplewidths = self.__buffer[2]

        binwidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = binwidths / (orders - 1.0)

        offset = 0
        for i, n in enumerate(orders):
            sample[offset : offset + n] = linspace(edges[i], edges[i + 1], n)
            weights[offset] = samplewidths[i] * 0.5
            if n > 2:
                weights[offset + 1 : offset + n - 2] = samplewidths[i]
            offset += n - 1
        weights[-1] = samplewidths[-1] * 0.5

        nodes[:] = (edges[1:] + edges[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return [outputs.iter_data()]

    def _fcn_gl1d(self, _, inputs, outputs) -> Optional[list]:
        """The 1d Gauss-Legendre sampling"""
        ordersX = outputs["ordersX"]
        edges = ordersX.dd.edges
        orders = ordersX.data
        nodes = self.__buffer[0]
        sample = outputs[0].data
        weights = outputs[1].data

        offset = 0
        for n in orders:
            if n >= 1:
                (
                    sample[offset : offset + n],
                    weights[offset : offset + n],
                ) = leggauss(n)
                # the `leggauss` works only with the range [-1,1],
                # so we need to transform the result to the original range
                sample[offset : offset + n] = (
                    0.5
                    * (sample[offset : offset + n] + 1)
                    * (edges[offset + n] - edges[offset])
                    + edges[offset]
                )
                weights[offset : offset + n] *= 0.5*(edges[offset + n] - edges[offset])
            offset += n

        nodes[:] = (edges[1:] + edges[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return [outputs.iter_data()]

    def _fcn_gl2d(self, _, inputs, outputs) -> Optional[list]:
        """The 2d Gauss-Legendre sampling"""
        # TODO: implement 2d GL sampling
        if self.debug:
            return [outputs.iter_data()]
