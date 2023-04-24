from typing import Optional, Sequence, Union

from numpy import array
from numpy.typing import ArrayLike, NDArray

from ..exception import InitializationError
from ..nodes import FunctionNode
from ..output import Output
from ..typefunctions import check_array_edges_consistency, check_edges_type


class Array(FunctionNode):
    """Creates a node with a single data output with predefined array"""

    _mode: str
    _data: NDArray
    _output = Output

    def __init__(
        self,
        name,
        arr,
        *,
        mode: str = "store",
        outname="array",
        mark: Optional[str] = None,
        edges: Optional[Union[Output, Sequence[Output]]] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._mode = mode
        if mark is not None:
            self._mark = mark
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

        if edges is not None:
            if isinstance(edges, Output):
                self._output.dd.axes_edges.append(edges)
            else:
                # assume that the edges are Sequence[Output]
                try:
                    self._output.dd.axes_edges.extend(edges)
                except Exception as exc:
                    raise InitializationError(
                        "Array: edges must be `Output` or `Sequence[Output]`, "
                        f"but given {edges=}, {type(edges)=}"
                    ) from exc

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
        check_array_edges_consistency(self, self._output) # checks dim and N+1 size

    def _post_allocate(self) -> None:
        if self._mode == "fill":
            return

        self._data = self._output._data

    def set(self, data: ArrayLike, check_taint: bool = False) -> bool:
        return self._output.set(data, check_taint)
