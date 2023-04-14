from typing import Literal, Optional

from numpy import (
    empty,
    errstate,
    integer,
    issubdtype,
    linspace,
    matmul,
    meshgrid,
    newaxis,
)
from numpy.polynomial.legendre import leggauss
from numpy.typing import DTypeLike, NDArray

from ..exception import InitializationError, TypeFunctionError
from ..nodes import FunctionNode
from ..typefunctions import check_input_dimension, check_inputs_number


def _gl_sampler(
    orders: NDArray, sample: NDArray, weights: NDArray, edges: NDArray
):
    """
    Uses `numpy.polynomial.legendre.leggauss` to sample points with weights
    on the range [-1,1] and transforms to any range [a, b]
    """
    offset = 0
    for i, n in enumerate(orders):
        if n < 1:
            continue
        (
            sample[offset : offset + n],
            weights[offset : offset + n],
        ) = leggauss(n)
        # transforms to the original range [a, b]
        sample[offset : offset + n] = 0.5 * (
            sample[offset : offset + n] * (edges[i + 1] - edges[i])
            + (edges[i + 1] + edges[i])
        )
        weights[offset : offset + n] *= 0.5 * (edges[i + 1] - edges[i])
        # NOTE: the operations above may allocate additional memory in runtime!
        offset += n


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
        self._add_output("x")
        if mode == "2d":
            self._add_input("ordersY", positional=False)
            self._add_output("y")
        self._add_output("weights", positional=False)
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
            shape = (lenX, lenY)
        else:
            shape = (lenX,)
        for output in (*self.outputs, self.outputs["weights"]):
            output.dd.shape = shape
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
        if self.mode == "rect":
            shapeX = (5, edgeshapeX)
        elif self.mode == "trap":
            shapeX = (2, edgeshapeX)
        elif self.mode == "gl":
            shapeX = (edgeshapeX,)
        else:
            lenY = sum(self.inputs["ordersY"].data)
            shapeY = (3, lenY)
            self.__bufferY = empty(shape=shapeY, dtype=self.dtype)
            lenX = sum(ordersX.data)
            shapeX = (3, lenX)
        self.__bufferX = empty(shape=shapeX, dtype=self.dtype)

    def _fcn_rect(self, _, inputs, outputs) -> Optional[list]:
        """The rectangular sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges  # n+1
        orders = ordersX.data  # n
        sample = outputs[0].data  # m = sum(orders)
        weights = outputs["weights"].data
        nodes = self.__bufferX[0]  # n
        binwidths = self.__bufferX[1]  # n
        samplewidths = self.__bufferX[2]  # n
        low = self.__bufferX[3]  # n
        high = self.__bufferX[4]  # n

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

        # NOTE: the operations below may allocate additional memory in runtime!
        nodes[:] = (edges[1:] + edges[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return list(outputs.iter_data())

    def _fcn_trap(self, _, inputs, outputs) -> Optional[list]:
        """The trapezoidal sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges  # n+1
        orders = ordersX.data  # n
        sample = outputs[0].data  # m = sum(orders)
        weights = outputs["weights"].data
        nodes = self.__bufferX[0]  # n
        samplewidths = self.__bufferX[1]  # n

        samplewidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = samplewidths[:] / (orders - 2.0)

        offset = 0
        for i, n in enumerate(orders):
            sample[offset : offset + n] = linspace(edges[i], edges[i + 1], n)
            weights[offset] = samplewidths[i] * 0.5
            if n > 2:
                weights[offset + 1 : offset + n - 2] = samplewidths[i]
            offset += n - 1
        weights[-1] = samplewidths[-1] * 0.5

        # NOTE: the operations below may allocate additional memory in runtime!
        nodes[:] = (edges[1:] + edges[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return list(outputs.iter_data())

    def _fcn_gl1d(self, _, inputs, outputs) -> Optional[list]:
        """The 1d Gauss-Legendre sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges
        orders = ordersX.data
        nodes = self.__bufferX
        sample = outputs[0].data
        weights = outputs["weights"].data

        _gl_sampler(orders, sample, weights, edges)

        # NOTE: the operations below may allocate additional memory in runtime!
        nodes[:] = (edges[1:] + edges[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = edges
            output.dd.axes_nodes = nodes

        if self.debug:
            return list(outputs.iter_data())

    def _fcn_gl2d(self, _, inputs, outputs) -> Optional[list]:
        """The 2d Gauss-Legendre sampling"""
        ordersX = inputs["ordersX"]
        ordersY = inputs["ordersY"]
        edgesX = ordersX.dd.axes_edges  # p + 1
        edgesY = ordersY.dd.axes_edges  # q + 1
        ordersX = ordersX.data
        ordersY = ordersY.data
        # NOTE: nodesX and nodesY need only p and q elements
        nodesX = self.__bufferX[0]  # (p, )
        nodesY = self.__bufferY[0]  # (q, )
        weightsX = self.__bufferX[1]  # (n, )
        weightsY = self.__bufferY[1]  # (m, )
        sampleX = self.__bufferX[2]  # (n, )
        sampleY = self.__bufferY[2]  # (m, )
        X = outputs[0].data  # (n, m)
        Y = outputs[1].data  # (n, m)
        weights = outputs["weights"].data  # (n, m)

        _gl_sampler(ordersX, sampleX, weightsX, edgesX)
        _gl_sampler(ordersY, sampleY, weightsY, edgesY)

        X[:], Y[:] = meshgrid(sampleX, sampleY, indexing="ij")
        matmul(weightsX[newaxis].T, weightsY[newaxis], out=weights)

        # NOTE: the operations below may allocate additional memory in runtime!
        p = edgesX.shape[0] - 1
        q = edgesY.shape[0] - 1
        nodesX[:p] = (edgesX[1:] + edgesX[:-1]) * 0.5
        nodesY[:q] = (edgesY[1:] + edgesY[:-1]) * 0.5
        for output in outputs:
            output.dd.axes_edges = [edgesX, edgesY]
            output.dd.axes_nodes = [nodesX[:p], nodesY[:q]]

        if self.debug:
            return list(outputs.iter_data())
