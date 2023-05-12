from typing import List, Optional
from numpy.typing import DTypeLike

from .types import EdgesLike, ShapeLike
from .labels import repr_pretty

class DataDescriptor:
    """
    The data descriptor class stores `dtype`, `shape`,
    `axes_edges` and `axes_nodes` information.
    """

    __slots__ = ("dtype", "shape", "axes_edges", "axes_nodes", "edges_inherited", "nodes_inherited")
    dtype: DTypeLike  # DTypeLike is already Optional
    shape: Optional[ShapeLike]
    axes_edges: Optional[List[EdgesLike]]
    axes_nodes: Optional[List[EdgesLike]]

    edges_inherited: bool
    nodes_inherited: bool

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

        self.edges_inherited = True
        self.nodes_inherited = True

    def __str__(self):
        return (f'{self.dtype} {self.shape}'
                f'{bool(self.axes_edges) and " [edges]" or ""}'
                f'{bool(self.axes_nodes) and " [nodes]" or ""}')

    _repr_pretty_ = repr_pretty

    @property
    def dim(self) -> int:
        """ Return the dimension of the data """
        return len(self.shape)
