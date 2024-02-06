from typing import Literal

from numpy import prod
from numpy.typing import DTypeLike
from numpy.typing import NDArray

from .labels import repr_pretty
from .types import EdgesLike
from .types import MeshesLike
from .types import ShapeLike


class DataDescriptor:
    """
    The data descriptor class stores `dtype`, `shape`,
    `axes_edges` and `axes_meshes` information.
    """

    __slots__ = ("dtype", "shape", "axes_edges", "axes_meshes", "edges_inherited", "meshes_inherited")
    dtype: DTypeLike  # DTypeLike is already Optional
    shape: ShapeLike | None
    axes_edges: EdgesLike
    axes_meshes: EdgesLike

    edges_inherited: bool
    meshes_inherited: bool

    def __init__(
        self,
        dtype: DTypeLike,  # DTypeLike is already Optional
        shape: ShapeLike | None,
        axes_edges: EdgesLike | None = None,
        axes_meshes: MeshesLike | None = None,
    ) -> None:
        """
        Sets the attributes
        """
        self.dtype = dtype
        self.shape = shape
        self.axes_edges = axes_edges or ()
        self.axes_meshes = axes_meshes or ()

        self.edges_inherited = True
        self.meshes_inherited = True

    def __str__(self):
        return (f'{self.dtype} {self.shape}'
                f'{bool(self.axes_edges) and " [edges]" or ""}'
                f'{bool(self.axes_meshes) and " [meshes]" or ""}')

    _repr_pretty_ = repr_pretty

    @property
    def dim(self) -> int:
        """ Return the dimension of the data """
        return len(self.shape)

    @property
    def size(self) -> int:
        """ Return the size of the data """
        return prod(self.shape)

    @property
    def edges_arrays(self) -> list[NDArray] | None:
        return tuple(o.data for o in self.axes_edges)

    @property
    def meshes_arrays(self) -> list[NDArray] | None:
        return tuple(o.data for o in self.axes_meshes)

    def axis_label(
        self,
        axis: int=0,
        axistype: Literal['any', 'edges', 'mesh']='any',
        *,
        root: bool=False
    ) -> str | None:
        if self.axes_edges and axistype in {'any', 'edges'}:
            try:
                if root:
                    return self.axes_edges[axis].labels.rootaxis
                else:
                    return self.axes_edges[axis].labels.axis
            except IndexError as e:
                raise RuntimeError(f'Invalid axis index {axis}') from e

        if self.axes_meshes and axistype in {'any', 'mesh'}:
            try:
                if root:
                    return self.axes_meshes[axis].labels.rootaxis
                else:
                    return self.axes_meshes[axis].labels.axis
            except IndexError as e:
                raise RuntimeError(f'Invalid axis index {axis}') from e

        return None
