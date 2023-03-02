from numpy.typing import DTypeLike

from .types import ShapeLikeT, EdgesLikeT


class Axis:
    """The axis class stores `edges` and `nodes`"""

    def __init__(
        self,
        edges: EdgesLikeT = None,
        nodes: EdgesLikeT = None,
    ) -> None:
        self._edges = edges
        self._nodes = nodes

    @property
    def edges(self) -> EdgesLikeT:
        return self._edges

    @property
    def nodes(self) -> EdgesLikeT:
        return self._nodes


class DataDescriptor:
    """The data descriptor class stores `dtype`, `shape` and `axis` information"""

    def __init__(
        self,
        dtype: DTypeLike = None,
        shape: ShapeLikeT = None,
        axis_edges: EdgesLikeT = None,
        axis_nodes: EdgesLikeT = None,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._axis = Axis(axis_edges, axis_nodes)

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def shape(self) -> ShapeLikeT:
        return self._shape

    @property
    def axis(self) -> Axis:
        return self._axis
