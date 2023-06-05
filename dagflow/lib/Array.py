from typing import Optional, Sequence, Union
from numbers import Number

from numpy import array, full
from numpy.typing import ArrayLike, NDArray, DTypeLike

from ..exception import InitializationError
from ..nodes import FunctionNode
from ..node import Node
from ..output import Output
from ..typefunctions import check_array_edges_consistency, check_edges_type

class Array(FunctionNode):
    """Creates a node with a single data output with predefined array"""
    __slots__ = ('_mode', '_data', '_output')

    _mode: str
    _data: NDArray
    _output: Output

    def __init__(
        self,
        name,
        arr,
        *,
        mode: str = "store",
        outname="array",
        mark: Optional[str] = None,
        edges: Union[Output, Sequence[Output], Node, None] = None,
        meshes: Union[Output, Sequence[Output], Node, None] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._mode = mode
        if mark is not None:
            self._labels.setdefault('mark', mark)
        self._data = array(arr, copy=True)

        if mode == "store":
            self._output = self._add_output(outname, data=self._data)
        elif mode == "store_weak":
            self._output = self._add_output(
                outname, data=self._data, owns_buffer=False
            )
        elif mode == "fill":
            self._output = self._add_output(
                outname, dtype=self._data.dtype, shape=self._data.shape
            )
        else:
            raise InitializationError(
                f'Array: invalid mode "{mode}"', node=self
            )

        self._functions.update(
            {
                "store": self._fcn_store,
                "store_weak": self._fcn_store,
                "fill": self._fcn_fill,
            }
        )
        self.fcn = self._functions[self._mode]

        self._init_edges(edges)
        self._init_meshes(meshes)

        if mode == "store":
            self.close()

    def _fcn_store(self, *args):
        return self._data

    def _fcn_fill(self, *args):
        data = self._output._data
        data[:] = self._data
        return data

    def _typefunc(self) -> None:
        check_edges_type(self, slice(None), "array") # checks List[Output]
        check_array_edges_consistency(self, "array") # checks dim and N+1 size

    def _post_allocate(self) -> None:
        if self._mode == "fill":
            return

        self._data = self._output._data

    def set(self, data: ArrayLike, check_taint: bool = False) -> bool:
        return self._output.set(data, check_taint)

    @classmethod
    def from_value(
        cls,
        name,
        value: Number,
        *,
        edges: Union[Output, Sequence[Output], Node],
        dtype: DTypeLike=None,
        **kwargs
    ):
        if isinstance(edges, Output):
            shape=(edges.dd.shape[0]-1,)
        elif isinstance(edges, Node):
            output = edges.outputs[0]
            shape=(output.dd.shape[0]-1,)
        elif isinstance(edges, Sequence):
            shape = tuple(output.dd.shape[0]-1 for output in edges)
        else:
            raise RuntimeError("Invalid edges specification")
        array = full(shape, value, dtype=dtype)
        return cls.make_stored(name, array, edges=edges, **kwargs)

    def _init_edges(self, edges: Union[Output, Sequence[Output], Node]):
        if not edges:
            return

        if not isinstance(edges, Sequence):
            edges = (edges,)

        dd = self._output.dd
        dd.edges_inherited = False
        for edgesi in edges:
            if isinstance(edgesi, Output):
                dd.axes_edges+=(edgesi,)
            elif isinstance(edgesi, Node):
                dd.axes_edges+=(edgesi.outputs[0],)
            else:
                raise InitializationError(
                    "Array: edges must be `Output/Node` or `Sequence[Output/Node]`, "
                    f"but given {edges=}, {type(edges)=}"
                )

    def _init_meshes(self, meshes: Union[Output, Sequence[Output], Node]):
        if not meshes:
            return

        if not isinstance(meshes, Sequence):
            meshes = (meshes,)

        dd = self._output.dd
        dd.meshes_inherited = False
        for meshesi in meshes:
            if isinstance(meshesi, Output):
                dd.axes_meshes+=(meshesi,)
            elif isinstance(meshesi, Node):
                dd.axes_meshes+=(meshesi.outputs[0],)
            else:
                raise InitializationError(
                    "Array: meshes must be `Output/Node` or `Sequence[Output/Node]`, "
                    f"but given {meshes=}, {type(meshes)=}"
                )
