# TODO: Implementing of 21GL

from typing import Literal, Optional

from numpy import (
    empty,
    errstate,
    integer,
    issubdtype,
    linspace,
    matmul,
    newaxis,
)
from numpy.polynomial.legendre import leggauss
from numpy.typing import DTypeLike, NDArray

from ..exception import InitializationError, TypeFunctionError
from ..nodes import FunctionNode
from ..typefunctions import check_input_dimension, check_inputs_number


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

    __bufferX: NDArray
    __bufferY: NDArray
    __bufferExtra: NDArray

    def __init__(
        self,
        *args,
        mode: Literal["rect", "trap", "gl", "2d"],
        dtype: DTypeLike = "d",
        align: Optional[Literal["left", "center", "right"]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if mode not in {"rect", "trap", "gl", "2d"}:
            raise InitializationError(
                f"Argument `mode` must be 'rect', 'trap', 'gl', or '2d', but given '{mode}'!",
                node=self,
            )
        if align is not None and mode != "rect":
            raise InitializationError(
                "Argument 'align' is used only within 'rect' mode!", node=self
            )
        self._dtype = dtype
        self._mode = mode
        self._align = align if align is not None else "center"
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
    def align(self) -> Optional[str]:
        return self._align

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an integration algorithm,
        determines dtype and shape for outputs
        """
        check_inputs_number(self, 0)
        lenX = self.__check_orders("ordersX")
        if self.mode == "2d":
            lenY = self.__check_orders("ordersY")
            self.outputs[0].dd.shape = (2, lenX, lenY)
            self.outputs[1].dd.shape = (lenX, lenY)
        else:
            shape = (lenX,)
            for output in self.outputs:
                output.dd.shape = shape
        for output in self.outputs:
            output.dd.dtype = self.dtype
        self.fcn = self._functions[self.mode]

    def __check_orders(self, name: str) -> int:
        """
        The method checks dimension (==1) of the input `name`, type (==`integer`),
        and returns the `dd.shape[0]`
        """
        check_input_dimension(self, name, 1)
        orders = self.inputs[name]
        if not issubdtype(orders.dd.dtype, integer):
            raise TypeFunctionError(
                f"The `name` must be array of integers, but given '{orders.dd.dtype}'!",
                node=self,
                input=orders,
            )
        return sum(orders.data)

    def _post_allocate(self) -> None:
        """Allocates the `buffer`, which elements are the follows:
        * 0: axis_nodes
        * 1: binwidths
        * 2: samplewidths
        * 3: low edges (only for `rect`)
        * 4: high edges (only for `rect`)
        """
        ordersX = self.inputs["ordersX"]
        edgeshapeX = ordersX.dd.axes_edges.shape[0] - 1
        npoints = sum(ordersX.data)
        if self.mode == "rect":
            shapeX = (5, edgeshapeX)
        elif self.mode == "trap":
            shapeX = (3, edgeshapeX)
        elif self.mode == "gl":
            shapeX = (edgeshapeX,)
        else:
            edgeshapeY = self.inputs["ordersY"].dd.axes_edges.shape[0] - 1
            shapeX = (3, edgeshapeX)
            shapeY = (3, edgeshapeY)
            self.__bufferY = empty(shape=shapeY, dtype=self.dtype)
        self.__bufferX = empty(shape=shapeX, dtype=self.dtype)

    def _fcn_rect(self, _, inputs, outputs) -> Optional[list]:
        """The rectangular sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges # n
        orders = ordersX.data # n
        sample = outputs[0].data # m = sum(orders)
        weights = outputs[1].data # m = sum(orders)
        nodes = self.__bufferX[0] # n
        binwidths = self.__bufferX[1] # n
        samplewidths = self.__bufferX[2] # n
        low = self.__bufferX[3] # n
        high = self.__bufferX[4] # n

        binwidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = binwidths / orders
        if self.align == "left":
            low[:] = edges[:-1]
            high[:] = edges[1:] - samplewidths
        elif self.align == "center":
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
            return list(outputs.iter_data())

    def _fcn_trap(self, _, inputs, outputs) -> Optional[list]:
        """The trapezoidal sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges
        orders = ordersX.data
        sample = outputs[0].data
        weights = outputs[1].data
        nodes = self.__bufferX[0]
        binwidths = self.__bufferX[1]
        samplewidths = self.__bufferX[2]

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
            return list(outputs.iter_data())

    def _fcn_gl1d(self, _, inputs, outputs) -> Optional[list]:
        """The 1d Gauss-Legendre sampling"""
        ordersX = outputs["ordersX"]
        edges = ordersX.dd.axes_edges
        orders = ordersX.data
        nodes = self.__bufferX[0]
        sample = outputs[0].data
        weights = outputs[1].data

        offset = 0
        for n in orders:
            if n < 1:
                continue
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
            weights[offset : offset + n] *= 0.5 * (
                edges[offset + n] - edges[offset]
            )
            offset += n

        nodes[:] = (edges[1:] + edges[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return list(outputs.iter_data())

    def _fcn_gl2d(self, _, inputs, outputs) -> Optional[list]:
        """The 2d Gauss-Legendre sampling"""
        ordersX = outputs["ordersX"]
        ordersY = outputs["ordersY"]
        edgesX = ordersX.dd.axes_edges
        edgesY = ordersY.dd.axes_edges
        ordersX = ordersX.data
        ordersY = ordersY.data
        nodesX = self.__bufferX[0]  # (n, )
        nodesY = self.__bufferY[0]  # (m, )
        weightsX = self.__bufferX[1]  # (n, )
        weightsY = self.__bufferY[1]  # (m, )
        sampleX = self.__bufferX[2]  # (n, )
        sampleY = self.__bufferY[2]  # (m, )
        sample = outputs[0].data  # (2, n, m)
        weights = outputs[1].data  # (n, m)

        offsetX = 0
        for n in ordersX:
            if n < 1:
                continue
            (
                sampleX[offsetX : offsetX + n],
                weightsX[offsetX : offsetX + n],
            ) = leggauss(n)
            # the `leggauss` works only with the range [-1,1],
            # so we need to transform the result to the original range
            sampleX[offsetX : offsetX + n] = (
                0.5
                * (sampleX[offsetX : offsetX + n] + 1)
                * (edgesX[offsetX + n] - edgesX[offsetX])
                + edgesX[offsetX]
            )
            weightsX[offsetX : offsetX + n] *= 0.5 * (
                edgesX[offsetX + n] - edgesX[offsetX]
            )
            offsetX += n

        offsetY = 0
        for n in ordersY:
            if n < 1:
                continue
            (
                sampleY[offsetY : offsetY + n],
                weightsY[offsetY : offsetY + n],
            ) = leggauss(n)
            # the `leggauss` works only with the range [-1,1],
            # so we need to transform the result to the original range
            sampleY[offsetY : offsetY + n] = (
                0.5
                * (sampleY[offsetY : offsetY + n] + 1)
                * (edgesY[offsetY + n] - edgesY[offsetY])
                + edgesY[offsetY]
            )
            weightsY[offsetY : offsetY + n] *= 0.5 * (
                edgesY[offsetY + n] - edgesY[offsetY]
            )
            offsetY += n

        for i, x in enumerate(sampleX):
            for j in range(len(sampleX[0, i])):
                sample[0, i, j] = x
        for i in range(len(sampleY[0])):
            sample[1, i, :] = sampleY
        matmul(weightsX[newaxis].T, weightsY, out=weights)

        nodesX[:] = (edgesX[1:] + edgesX[:-1]) * 0.5
        nodesY[:] = (edgesY[1:] + edgesY[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = [edgesX, edgesY]
            output.dd.axes_nodes = [nodesX, nodesY]

        if self.debug:
            return list(outputs.iter_data())
