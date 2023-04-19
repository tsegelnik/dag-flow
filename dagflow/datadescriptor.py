from typing import List, Optional

from numpy.typing import DTypeLike

from .types import EdgesLike, ShapeLike


class DataDescriptor:
    """
    The data descriptor class stores `dtype`, `shape`,
    `axes_edges` and `axes_nodes` information.
    """

    __slots__ = ("dtype", "shape", "axes_edges", "axes_nodes")
    dtype: DTypeLike  # DTypeLike is already Optional
    shape: Optional[ShapeLike]
    axes_edges: Optional[List[EdgesLike]]
    axes_nodes: Optional[List[EdgesLike]]

    def __init__(
        self,
        dtype: DTypeLike,  # DTypeLike is already Optional
        shape: Optional[ShapeLike],
        axes_edges: Optional[List[EdgesLike]] = None,
        axes_nodes: Optional[List[EdgesLike]] = None,
    ) -> None:
        """
        Sets the attributes
        """
        self.dtype = dtype
        self.shape = shape
        self.axes_edges = axes_edges or []
        self.axes_nodes = axes_nodes or []

    @property
    def dim(self) -> int:
        """ Return the dimension of the data """
        return len(self.shape)
