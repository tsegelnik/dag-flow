from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING

from numpy import nan, ndarray
from ordered_set import OrderedSet

from nestedmapping import NestedMapping
from nestedmapping.typing import Key, KeyLike, TupleKey, strkey
from nestedmapping.visitor import NestedMappingVisitor

from ..tools.logger import DEBUG, INFO1, INFO3, logger
from .input import Input
from .node import Node
from .output import Output

if TYPE_CHECKING:
    from collections.abc import Container, Mapping, MutableSet
    from typing import Any, Literal

    from matplotlib.axes import Axes
    from pandas import DataFrame

# TODO: Maybe this import and set options should be in some function?
#       Set options in the current file may lead to unexpected results in other places
from pandas import set_option as pandas_set_option

pandas_set_option("display.max_rows", None)
pandas_set_option("display.max_colwidth", 100)


def trunc(text: str, width: int) -> str:
    return "\n".join(line[:width] for line in text.split("\n"))


def _fillna(df: DataFrame, columnname: str, replacement: str):
    column = df[columnname]
    if not column.isnull().values.any():
        return

    # if column.dtype!='O':
    #     df[columnname] = (column:=column.astype('O', copy=False))

    newcol = column.fillna(replacement)
    df[columnname] = newcol


class NodeStorage(NestedMapping):
    __slots__ = ("_remove_connected_inputs",)
    _remove_connected_inputs: bool

    def __init__(
        self,
        *args,
        remove_connected_inputs: bool = True,
        default_containers: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("sep", ".")
        kwargs.setdefault("recursive_to_others", False)
        super().__init__(*args, **kwargs)

        self._remove_connected_inputs = remove_connected_inputs

        if not default_containers:
            return
        for name in ("parameters", "stat", "nodes", "data", "inputs", "outputs"):
            self.create_child(name)

    def plot(self, *args, **kwargs) -> None:
        self.visit(PlotVisitor(*args, **kwargs))

    def savegraphs(
        self,
        folder: str,
        mindepth: int = -2,
        maxdepth: int = 2,
        accept_index: Mapping[str, str | int | Container[str | int]] | None = None,
        **kwargs,
    ):
        from os import makedirs

        from ..parameters import Parameters
        from ..plot.graphviz import GraphDot

        items = list(self.walkitems())
        nitems = len(items)
        folder0 = folder
        for i, (key, node) in enumerate(items):
            match node:
                case Node():
                    pass
                case Parameters():
                    if constraint := node.constraint:
                        node = constraint._norm_node
                    else:
                        node = node._value_node
                case Output():
                    node = node.node
                case _:
                    continue
            try:
                if not node.labels.index_in_mask(accept_index):
                    continue
            except AttributeError:
                continue

            stem, index = [], []
            index_values = node.labels.index_values
            for skey in key:
                if skey in index_values:
                    index.append(skey)
                else:
                    stem.append(skey)
            if stem:
                stem, index = stem[:-1], stem[-1:] + index

            folder = f"{folder0}/{'/'.join(stem).replace('.', '_')}"
            filename = "_".join(index).replace(".", "_")
            makedirs(folder, exist_ok=True)
            fullname = f"{folder}/{filename}.dot"

            gd = GraphDot.from_nodes([node], mindepth=mindepth, maxdepth=maxdepth, **kwargs)
            gd.savegraph(fullname, quiet=True)

            logger.log(INFO1, f"Write: {fullname} [{i + 1}/{nitems}]")

    def __setitem__(self, key: KeyLike, item: Any) -> None:
        from ..parameters import Parameter, Parameters

        match item:
            case Node() | Output() | Parameter() | Parameters():
                logger.log(INFO3, f"Set {self.joinkey(key)}")
            case Input():
                logger.log(DEBUG, f"Set {self.joinkey(key)}")

        super().__setitem__(key, item)

    #
    # Connectors
    #
    def __rshift__(self, other: NestedMapping):
        """`self >> other`

        The connection is allowed only with `NestedMapping`.
        It is done within strict matching (by names) of objects in the two dicts.
        """
        if not isinstance(other, NestedMapping):
            raise RuntimeError("Operator >> RHS should be NestedMapping")

        from nestedmapping import walkkeys
        from nestedmapping.tools import match_keys

        keys_left = list(walkkeys(self))
        keys_right = walkkeys(other)

        nconnections = 0
        to_remove = []

        def function(i: int, outkey: TupleKey, inkey: TupleKey):
            nonlocal nconnections
            out = self[outkey]
            inp = other[inkey]

            try:
                out >> inp
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid NodeStorage>> types for {outkey}/{inkey}: {type(out)}/{type(inp)}"
                ) from e

            if self._remove_connected_inputs and isinstance(inp, (Input, tuple)):
                to_remove.append(inkey)

            nconnections += 1

        match_keys((keys_left,), keys_right, function, left_in_right=True, right_in_left=False)

        if nconnections == 0:
            raise ConnectionError("No connections are done")

        for key in to_remove:
            del other[key]

    def __lshift__(self, other: NestedMapping):
        """`self << other`

        Such connection iterates over child objects and attemps
        to use `<<` operator implemented inside them. Usually, such child objects are `Node`,
        where non-strict pattern matching is used for connection.
        """
        if not isinstance(other, NestedMapping):
            raise RuntimeError("Operator >> RHS should be nestedmapping")

        for keyleft, node in self.walkitems():
            try:
                node << other
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid NodeStorage<< types for {keyleft}: {type(node)} {type(other)}"
                ) from e

    def __rlshift__(self, other):
        """`other << self`

        Such connection iterates over `other` and attemps to use `<<` operator
        for every element. Usually, such child objects are `Node`,
        where non-strict pattern matching is used for connection.
        """
        if not isinstance(other, (Sequence, Generator)):
            raise RuntimeError(f"Cannot connect `{type(other)} << NodeStorage`!")

        for obj in other:
            try:
                obj << self
            except Exception as exc:
                raise RuntimeError(f"Cannot connect `{type(obj)} << NodeStorage`!") from exc

    #
    # Finalizers
    #
    def read_paths(self, *, index: Mapping[str, Sequence[str]] = {}) -> None:
        for key, value in self.walkitems():
            labels = getattr(value, "labels", None)
            if labels is None:
                continue

            path = ".".join(key)
            labels.paths.append(path)

    def read_labels(
        self,
        source: NestedMapping | dict,
        *,
        strict: bool = False,
        processed_keys_set: MutableSet[Key] | None = None,
    ) -> None:
        source = NestedMapping(source, sep=".")

        if processed_keys_set is None:
            processed_keys_set = set()

        def get_label(key):
            try:
                nkey = key + ("node",)
                labels = source.get_dict(nkey)
            except (KeyError, TypeError):
                pass
            else:
                processed_keys_set.add(nkey)
                return labels, None

            try:
                labels = source.get_dict(key)
            except (KeyError, TypeError):
                pass
            else:
                processed_keys_set.add(key)
                return labels, None

            keyleft = list(key[:-1])
            keyright = [key[-1]]
            while keyleft:
                groupkey = keyleft + ["group"]
                try:
                    labels = source.get_dict(groupkey)
                except (KeyError, TypeError):
                    keyright.insert(0, keyleft.pop())
                else:
                    processed_keys_set.add(tuple(groupkey))
                    return labels, keyright

            return None, None

        for key, object in self.walkitems():
            if not isinstance(object, (Node, Output)):
                continue

            logger.log(DEBUG, f"Look up label for {'.'.join(key)}")
            labels, subkey = get_label(key)
            if labels is None:
                continue
            if isinstance(labels, NestedMapping):
                labels = labels.object
            logger.log(DEBUG, "... found")

            if isinstance(object, Node):
                object.labels.update(labels)
            elif isinstance(object, Output):
                if object.labels is object.node.labels and len(object.node.outputs) != 1:
                    object.labels = object.node.labels.copy()
                object.labels.update(labels)

            if subkey:
                skey = ".".join(subkey)
                object.labels.format(
                    index=subkey, key=skey, space_key=f" {skey}", key_space=f"{skey} "
                )

        if strict:
            for key in processed_keys_set:
                source.delete_with_parents(key)
            if source:
                raise RuntimeError(
                    f"The following label groups were not used: {tuple(source.keys())}"
                )

    def remove_connected_inputs(self, key: Key = ()):
        source = self(key)

        def connected(input: Input | tuple[Input, ...]):
            match input:
                case Input():
                    return input.connected()
                case tuple():
                    return all(inp.connected() for inp in input)

        to_remove = [key for key, input in source.walkitems() if connected(input)]
        for key in to_remove:
            source.delete_with_parents(key)
        for key, dct in tuple(source.walkdicts()):
            if not dct:
                source.delete_with_parents(key)

    #
    # Converters
    #
    def to_list(self, **kwargs) -> list:
        return self.visit(ParametersVisitor(kwargs)).data_list

    def to_dict(self, **kwargs) -> NestedMapping:
        return self.visit(ParametersVisitor(kwargs)).data_dict

    def to_df(self, *, columns: list[str] | None = None, **kwargs) -> DataFrame:
        from pandas import DataFrame

        dct = self.to_list(**kwargs)
        if columns is None:
            columns = [
                "path",
                "value",
                "central",
                "sigma",
                "flags",
                "count",
                "shape",
                "label",
            ]
        df = DataFrame(dct, columns=columns)

        df.insert(4, "sigma_rel_perc", df["sigma"])
        sigma_rel_perc = df["sigma"] / df["central"] * 100.0
        sigma_rel_perc[df["central"] == 0] = nan
        df["sigma_rel_perc"] = sigma_rel_perc

        for key in ("count", "shape", "value", "central", "sigma", "sigma_rel_perc"):
            if df[key].isna().all():
                del df[key]
            else:
                _fillna(df, key, "-")

        if "value" in df.columns:
            _fillna(df, "value", "-")

        for col in ("flags", "label", "count", "shape"):
            if col in df.columns:
                _fillna(df, col, "")

        if (df["flags"] == "").all():
            del df["flags"]

        return df

    def to_string(self, **kwargs) -> str:
        df = self.to_df()
        kwargs.setdefault("index", False)
        return df.to_string(**kwargs)

    def to_table(
        self,
        *,
        df_kwargs: Mapping = {},
        truncate: int | bool | Literal["auto"] = False,
        **kwargs,
    ) -> str:
        from shutil import get_terminal_size

        from tabulate import tabulate

        df = self.to_df(**df_kwargs)

        kwargs.setdefault("headers", df.columns)
        kwargs.setdefault("showindex", False)
        ret = tabulate(df, **kwargs)

        match truncate:
            case True:
                truncate = get_terminal_size().columns
            case False | int():
                pass
            case "auto":
                from sys import stdout

                truncate = get_terminal_size().columns if stdout.isatty() else False
            case _:
                raise RuntimeError(f"Invalid {truncate=} value")

        return trunc(ret, width=truncate) if truncate else ret

    def print(self, *args, **kwargs) -> None:
        print(self.to_table(*args, **kwargs))

    def to_text_file(self, filename: str, **kwargs):
        table = self.to_table(**kwargs)

        with open(filename, "w") as out:
            logger.log(INFO1, f"Write: {filename}")
            out.write(table)

    def to_latex_file(
        self, filename: str | None = None, *, return_df: bool = False, **kwargs
    ) -> str | tuple[str, DataFrame]:
        df = self.to_df(label_from="latex", **kwargs)
        tex = df.to_latex(escape=False)

        if filename:
            with open(filename, "w") as out:
                logger.log(INFO1, f"Write: {filename}")
                out.write(tex)

        return tex, df if return_df else tex

    def to_datax(self, filename: str, **kwargs) -> None:
        from LaTeXDatax import datax

        data = self.to_dict(**kwargs)
        include = ("value", "central", "sigma", "sigma_rel_perc")
        odict = {".".join(k): v for k, v in data.walkitems() if (k and k[-1] in include)}
        logger.log(INFO1, f"Write: {filename}")
        datax(filename, **odict)

    def to_root(self, filename: str) -> None:
        from ..export.to_root import ExportToRootVisitor

        visitor = ExportToRootVisitor(filename)
        self.visit(visitor)

    #
    # Current storage, context
    #
    @staticmethod
    def current() -> NodeStorage | None:
        return _context_storage[-1] if _context_storage else None

    def __enter__(self) -> NodeStorage:
        _context_storage.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _context_storage.pop() != self:
            raise RuntimeError("NodeStorage: invalid context exit")

    @classmethod
    def update_current(cls, storage: NestedMapping, *, strict: bool = True, verbose: bool = False):
        if (common_storage := cls.current()) is None:
            return

        if verbose:
            print("Update current storage with:")
            for key in storage.walkjoinedkeys():
                print(f"- {key}")

        if strict:
            common_storage ^= storage
        else:
            common_storage |= storage


_context_storage: list[NodeStorage] = []


class PlotVisitor(NestedMappingVisitor):
    __slots__ = (
        "_show_all",
        "_folder",
        "_format",
        "_args",
        "_kwargs",
        "_minimal_data_size",
        "_active_figures",
        "_overlay_priority",
        "_currently_active_overlay",
        "_close_on_exitdict",
    )
    _show_all: bool
    _folder: str | None
    _format: str | None
    _args: Sequence
    _kwargs: dict
    _minimal_data_size: int
    _active_figures: dict
    _overlay_priority: Sequence[OrderedSet]
    _currently_active_overlay: OrderedSet | None
    _close_on_exitdict: bool

    def __init__(
        self,
        *args,
        show_all: bool = False,
        folder: str | None = None,
        format: str = "pdf",
        minimal_data_size: int = 1,
        overlay_priority: Sequence[Sequence[str]] = ((),),
        **kwargs,
    ):
        self._show_all = show_all
        self._folder = folder
        self._format = format
        self._args = args
        self._kwargs = kwargs
        self._minimal_data_size =  minimal_data_size
        self._overlay_priority = tuple(OrderedSet(sq) for sq in overlay_priority)
        self._currently_active_overlay = None
        self._close_on_exitdict = False
        self._active_figures = {}

        if self._show_all:
            self._kwargs["show"] = False
            self._kwargs["close"] = False
        elif self._folder is not None:
            self._kwargs["close"] = False

    def _try_start_join(self, key: TupleKey) -> tuple[tuple[str, ...] | None, str | None, bool]:
        key_set = OrderedSet(key)
        need_new_figure = True
        if self._currently_active_overlay is None:
            for indices_set in self._overlay_priority:
                if match := indices_set.intersection(key_set):
                    self._currently_active_overlay = indices_set
                    self._close_on_exitdict = match[0] == key_set[-1]
                    break
            else:
                return key, None, True
        elif match := self._currently_active_overlay.intersection(key_set):
            need_new_figure = False
        else:
            self._currently_active_overlay = None
            return key, None, True

        key_major = key_set.difference(self._currently_active_overlay)
        return tuple(key_major), match[0], need_new_figure

    def _makefigure(
        self, key: TupleKey, *, force_new: bool = False
    ) -> tuple[Axes, tuple[str, ...] | None, str | None, bool]:
        from matplotlib.pyplot import sca, subplots

        def mkfig(storekey: TupleKey | None = None) -> Axes:
            fig, ax = subplots(1, 1)
            if storekey is not None:
                self._active_figures[tuple(storekey)] = fig
            return ax

        if force_new:
            return mkfig(), key, None, True

        index_major, index_minor, need_new_figure = self._try_start_join(key)
        if need_new_figure or (fig := self._active_figures.get(tuple(index_major))) is None:
            return mkfig(index_major), index_major, index_minor, True

        sca(ax := fig.axes[0])
        return ax, index_major, index_minor, False

    def _savefig(self, key: TupleKey, *, close: bool = True):
        from os import makedirs
        from os.path import dirname

        from matplotlib.pyplot import close as closefig
        from matplotlib.pyplot import savefig

        if self._folder:
            path = "/".join(key).replace(".", "_")
            filename = f"{self._folder}/{path}.{self._format}"
            makedirs(dirname(filename), exist_ok=True)

            logger.log(INFO1, f"Write: {filename}")
            savefig(filename)

        if close:
            closefig()

    def _close_figures(self):
        from matplotlib.pyplot import sca

        for key, fig in self._active_figures.items():
            sca(fig.axes[0])
            self._savefig(key, close=True)
        self._active_figures = {}

    def start(self, dct):
        pass

    def enterdict(self, key, v):
        pass

    def exitdict(self, key, v):
        if self._close_on_exitdict:
            self._close_figures()
            self._close_on_exitdict = False
            self._currently_active_overlay = None
            return

        if self._currently_active_overlay and not self._currently_active_overlay.intersection(key):
            self._close_figures()
            self._currently_active_overlay = None

    def visit(self, key, output):
        from ..plot.plot import plot_auto

        if not isinstance(output, Output) or not output.labels.plottable:
            logger.log(DEBUG, f"Do not plot {strkey(key)} of not supported type")
            return
        if not output.labels.plottable:
            logger.log(DEBUG, f"Do not plot {strkey(key)}, configured to be not plottable")
            return
        if output.dd.size<self._minimal_data_size:
            logger.log(DEBUG, f"Do not plot {strkey(key)} of size {output.dd.size}<{self._minimal_data_size}")
            return

        nd = output.dd.dim
        _, index_major, index_minor, newfig = self._makefigure(key, force_new=(nd == 2))

        kwargs = self._kwargs.copy()
        if index_minor:
            kwargs.setdefault("label", index_minor)
        kwargs.setdefault("show_path", newfig)

        plot_auto(output, *self._args, **kwargs)
        if not index_minor:
            self._savefig(key, close=True)

    def stop(self, dct):
        from matplotlib.pyplot import show

        if self._show_all:
            show()
        self._close_figures()


class ParametersVisitor(NestedMappingVisitor):
    __slots__ = ("_kwargs", "_data_list", "_localdata", "_path")
    _kwargs: dict
    _data_list: list[dict]
    _data_dict: NestedMapping
    _localdata: list[dict]
    _paths: list[tuple[str, ...]]
    _path: tuple[str, ...]
    # _npars: List[int]

    def __init__(self, kwargs: dict):
        self._kwargs = kwargs
        # self._npars = []

    @property
    def data_list(self) -> list[dict]:
        return self._data_list

    @property
    def data_dict(self) -> NestedMapping:
        return self._data_dict

    def start(self, dct):
        self._data_list = []
        self._data_dict = NestedMapping({}, sep=".")
        self._path = ()
        self._paths = []
        self._localdata = []

    def enterdict(self, k, v):
        self._store()
        self._path = k
        self._paths.append(self._path)
        self._localdata = []

    def visit(self, key, value):
        match value:
            case ndarray():
                dct = {"shape": value.shape, "label": "data"}
            case tuple() | list():
                dct = {"count": len(value), "label": "sequence"}
            case _:
                try:
                    dct = value.to_dict(**self._kwargs)
                except (AttributeError, IndexError):
                    return

        if dct is None:
            return

        subkey = key[len(self._path) :]
        subkeystr = ".".join(subkey)

        dct["path"] = self._path and f".. {subkeystr}" or subkeystr
        self._localdata.append(dct)

        self._data_dict[key] = dct

    def exitdict(self, k, v):
        self._store()

        if self._paths:
            del self._paths[-1]

            self._path = self._paths[-1] if self._paths else ()

    def _store(self):
        if not self._localdata:
            return

        self._data_list.append(
            {
                "path": f"group: {'.'.join(self._path)} [{len(self._localdata)}]",
                "count": len(self._localdata),
                "label": "[group]",
            }
        )
        self._data_list.extend(self._localdata)
        self._localdata = []

    def stop(self, dct):
        pass
