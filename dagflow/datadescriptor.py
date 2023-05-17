from typing import List, Optional
from numpy.typing import DTypeLike, NDArray
from numpy import product

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
    axes_edges: EdgesLike
    axes_nodes: EdgesLike

    edges_inherited: bool
    nodes_inherited: bool

    def __init__(
        self,
        dtype: DTypeLike,  # DTypeLike is already Optional
        shape: Optional[ShapeLike],
        axes_edges: EdgesLike = None,
        axes_nodes: EdgesLike = None,
    ) -> None:
        """
        Sets the attributes
        """
        self.dtype = dtype
        self.shape = shape
        self.axes_edges = axes_edges or ()
        self.axes_nodes = axes_nodes or ()

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

    @property
    def size(self) -> int:
        """ Return the size of the data """
        return product(self.shape)

    @property
    def edges_arrays(self) -> Optional[List[NDArray]]:
        return tuple(o.data for o in self.axes_edges)

    @property
    def nodes_arrays(self) -> Optional[List[NDArray]]:
        return tuple(o.data for o in self.axes_nodes)

    def axis_label(self, axis: int=0, axistype: str='any', *, fallback='text') -> str:
        if self.axes_edges and axistype in {'any', 'edges'}:
            try:
                return self.axes_edges[axis].node.label('axis', fallback=fallback)
            except IndexError as e:
                raise RuntimeError(f'Invalid axis index {axis}') from e

        if self.axes_nodes and axistype in {'any', 'nodes'}:
            try:
                return self.axes_nodes[axis].node.label('axis', fallback=fallback)
            except IndexError as e:
                raise RuntimeError(f'Invalid axis index {axis}') from e

        return ''
