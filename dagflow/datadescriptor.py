from typing import Optional, Tuple
from numpy.typing import DTypeLike

from .types import ShapeLike, EdgesLike


class DataDescriptor:
    """
    The data descriptor class stores `dtype`, `shape`,
    `axes_edges` and `axes_nodes` information.
    """

    __slots__ = ("dtype", "shape", "axes_edges", "axes_nodes")
    dtype: DTypeLike # DTypeLike is already Optional
    shape: Optional[ShapeLike]
    axes_edges: Optional[Tuple[EdgesLike]]
    axes_nodes: Optional[Tuple[EdgesLike]]

    def __init__(
        self,
        dtype: DTypeLike, # DTypeLike is already Optional
        shape: Optional[ShapeLike],
        axes_edges: Optional[Tuple[EdgesLike]] = None,
        axes_nodes: Optional[Tuple[EdgesLike]] = None,
    ) -> None:
        """
        Sets the attributes
        """
        self.dtype = dtype
        self.shape = shape
        self.axes_edges = axes_edges
        self.axes_nodes = axes_nodes
