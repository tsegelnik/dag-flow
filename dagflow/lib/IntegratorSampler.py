from typing import Literal, Optional

from numpy import (
    empty,
    errstate,
    integer,
    linspace,
    matmul,
    meshgrid,
    newaxis,
)
from numpy.polynomial.legendre import leggauss
from numpy.typing import DTypeLike, NDArray

from ..exception import InitializationError
from ..nodes import FunctionNode
from ..typefunctions import (
    check_input_dimension,
    check_input_edges_dim,
    check_inputs_number,
    check_output_subtype,
)


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

    __slots__ = ("__bufferX", "__bufferY")

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
        lenX, edgesX = self.__check_orders("ordersX")
        if self.mode == "2d":
            lenY, edgesY = self.__check_orders("ordersY")
            shape = (lenX, lenY)
            edges = [edgesX, edgesY]
        else:
            shape = (lenX,)
            edges = [edgesX]
        for output in (*self.outputs, self.outputs["weights"]):
            output.dd.dtype = self.dtype
            output.dd.shape = shape
            output.dd.axes_edges = edges
        self.fcn = self._functions[self.mode]

    def __check_orders(self, name: str) -> tuple:
        """
        The method checks dimension (==1) of the input `name`, type (==`integer`),
        and returns the `dd.shape[0]`
        """
        check_input_dimension(self, name, 1)
        orders = self.inputs[name]
        check_output_subtype(self, orders, integer)
        check_input_edges_dim(self, name, 1)
        return sum(orders.data), orders.dd.axes_edges[0]

    def _post_allocate(self) -> None:
        """Allocates the `buffer`"""
        ordersX = self.inputs["ordersX"]
        edgeshapeX = ordersX.dd.axes_edges[0].dd.shape[0] - 1
        if self.mode == "rect":
            shapeX = (4, edgeshapeX)
        elif self.mode in {"trap", "gl"}:
            shapeX = (edgeshapeX,)
        else:
            lenY = sum(self.inputs["ordersY"].data)
            shapeY = (2, lenY)
            self.__bufferY = empty(shape=shapeY, dtype=self.dtype)
            lenX = sum(ordersX.data)
            shapeX = (2, lenX)
        self.__bufferX = empty(shape=shapeX, dtype=self.dtype)

    def _fcn_rect(self, _, inputs, outputs) -> Optional[list]:
        """The rectangular sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges[0]._data  # n+1
        orders = ordersX.data  # n
        sample = outputs[0].data  # m = sum(orders)
        weights = outputs["weights"].data
        binwidths = self.__bufferX[0]  # n
        samplewidths = self.__bufferX[1]  # n
        low = self.__bufferX[2]  # n
        high = self.__bufferX[3]  # n

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

        if self.debug:
            return list(outputs.iter_data())

    def _fcn_trap(self, _, inputs, outputs) -> Optional[list]:
        """The trapezoidal sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges[0]._data  # n+1
        orders = ordersX.data  # n
        sample = outputs[0].data  # m = sum(orders)
        weights = outputs["weights"].data
        samplewidths = self.__bufferX  # n

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

        if self.debug:
            return list(outputs.iter_data())

    def _fcn_gl1d(self, _, inputs, outputs) -> Optional[list]:
        """The 1d Gauss-Legendre sampling"""
        ordersX = inputs["ordersX"]
        edges = ordersX.dd.axes_edges[0]._data
        orders = ordersX.data
        sample = outputs[0].data
        weights = outputs["weights"].data

        _gl_sampler(orders, sample, weights, edges)

        if self.debug:
            return list(outputs.iter_data())

    def _fcn_gl2d(self, _, inputs, outputs) -> Optional[list]:
        """The 2d Gauss-Legendre sampling"""
        ordersX = inputs["ordersX"]
        ordersY = inputs["ordersY"]
        edgesX = ordersX.dd.axes_edges[0]._data  # p + 1
        edgesY = ordersY.dd.axes_edges[0]._data  # q + 1
        ordersX = ordersX.data
        ordersY = ordersY.data
        weightsX = self.__bufferX[0]  # (n, )
        weightsY = self.__bufferY[0]  # (m, )
        sampleX = self.__bufferX[1]  # (n, )
        sampleY = self.__bufferY[1]  # (m, )
        X = outputs[0].data  # (n, m)
        Y = outputs[1].data  # (n, m)
        weights = outputs["weights"].data  # (n, m)

        _gl_sampler(ordersX, sampleX, weightsX, edgesX)
        _gl_sampler(ordersY, sampleY, weightsY, edgesY)

        X[:], Y[:] = meshgrid(sampleX, sampleY, indexing="ij")
        matmul(weightsX[newaxis].T, weightsY[newaxis], out=weights)

        if self.debug:
            return list(outputs.iter_data())
