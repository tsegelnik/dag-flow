from __future__ import annotations

from typing import TYPE_CHECKING

from orderedset import OrderedSet

from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.typing import Key, KeyLike, TupleKey
from multikeydict.visitor import NestedMKDictVisitor

from .input import Input
from .logger import DEBUG, INFO1, INFO3, logger
from .node import Node
from .output import Output

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from typing import TYPE_CHECKING, Any, Literal

    from collections.abc import Mapping, MutableSet, Sequence

from LaTeXDatax import datax
from numpy import nan, ndarray
from pandas import DataFrame
from pandas import set_option as pandas_set_option
from tabulate import tabulate

pandas_set_option("display.max_rows", None)
pandas_set_option("display.max_colwidth", 100)

from shutil import get_terminal_size


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

class NodeStorage(NestedMKDict):
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
        for name in ("parameter", "stat", "nodes", "data", "inputs", "outputs"):
            self.child(name)

    def plot(self, *args, **kwargs) -> None:
        self.visit(PlotVisitor(*args, **kwargs))

    def __setitem__(self, key: KeyLike, item: Any) -> None:
        from .parameters import Parameter, Parameters
        match item:
            case Node() | Output() | Parameter() | Parameters():
                logger.log(INFO3, f"Set {self.joinkey(key)}")
            case Input():
                logger.log(DEBUG, f"Set {self.joinkey(key)}")

        super().__setitem__(key, item)

    #
    # Connectors
    #
    def __rshift__(self, other: NestedMKDict):
        if not isinstance(other, NestedMKDict):
            raise RuntimeError("Operator >> RHS should be NestedMKDict")

        from multikeydict.nestedmkdict import walkkeys
        from multikeydict.tools import match_keys

        keys_left = list(walkkeys(self))
        keys_right = walkkeys(other)

        nconnections = 0
        to_remove = []

        def fcn(i: int, outkey: TupleKey, inkey: TupleKey):
            nonlocal nconnections
            output = self[outkey]
            input = other[inkey]

            try:
                output >> input
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid NodeStorage>> types for {outkey}/{inkey}: {type(output)}/{type(input)}"
                ) from e

            if self._remove_connected_inputs and isinstance(input, Input):
                to_remove.append(inkey)

            nconnections += 1

        match_keys((keys_left,), keys_right, fcn, left_in_right=True, right_in_left=False)

        if nconnections == 0:
            raise ConnectionError("No connections are done")

        for key in to_remove:
            del other[key]

    def __lshift__(self, other: NestedMKDict):
        if not isinstance(other, NestedMKDict):
            raise RuntimeError("Operator >> RHS should be NestedMKDict")

        for keyleft, node in self.walkitems():
            try:
                node << other
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid NodeStorage<< types for {keyleft}: {type(node)} {type(other)}"
                ) from e

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
        source: NestedMKDict | dict,
        *,
        strict: bool = False,
        processed_keys_set: MutableSet[Key] | None = None,
    ) -> None:
        source = NestedMKDict(source, sep=".")

        if processed_keys_set is None:
            processed_keys_set = set()

        def get_label(key):
            try:
                # if strict:
                #     labels = source.pop(key, delete_parents=True)
                # else:
                labels = source(key)
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
                    labels = source(groupkey)
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
            logger.log(DEBUG, "... found")

            if isinstance(object, Node):
                object.labels.update(labels)
            elif isinstance(object, Output):
                if (
                    object.labels is object.node.labels
                    and len(object.node.outputs) != 1
                ):
                    object.labels = object.node.labels.copy()
                object.labels.update(labels)

            if subkey:
                skey = ".".join(subkey)
                object.labels.format(space_key=f" {skey}", key_space=f"{skey} ")

        if strict:
            for key in processed_keys_set:
                source.delete_with_parents(key)
            if source:
                raise RuntimeError(
                    f"The following label groups were not used: {tuple(source.keys())}"
                )

    def remove_connected_inputs(self, key: Key = ()):
        source = self(key)
        def connected(input: Input | tuple[Input,...]):
            match input:
                case Input():
                    return input.connected()
                case tuple():
                    return all(inp.connected() for inp in input)

        to_remove = [
            key
            for key, input in source.walkitems()
            if connected(input)
        ]
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

    def to_dict(self, **kwargs) -> NestedMKDict:
        return self.visit(ParametersVisitor(kwargs)).data_dict

    def to_df(self, *, columns: list[str] | None = None, **kwargs) -> DataFrame:
        dct = self.to_list(**kwargs)
        if columns is None:
            columns = ["path", "value", "central", "sigma", "flags", "shape", "label"]
        df = DataFrame(dct, columns=columns)

        df.insert(4, "sigma_rel_perc", df["sigma"])
        sigma_rel_perc = df["sigma"] / df["central"] * 100.0
        sigma_rel_perc[df["central"]==0] = nan
        df["sigma_rel_perc"] = sigma_rel_perc

        for key in ("central", "sigma", "sigma_rel_perc"):
            if df[key].isna().all():
                del df[key]
            else:
                _fillna(df, key, "-")

        _fillna(df, "value", "-")
        _fillna(df, "flags", "")
        _fillna(df, "label", "")
        _fillna(df, "shape", "")

        if (df["flags"] == "").all():
            del df["flags"]

        return df

    def to_string(self, **kwargs) -> str:
        df = self.to_df()
        return df.to_string(**kwargs)

    def to_table(
        self, *, df_kwargs: Mapping = {}, truncate: int | bool | Literal["auto"] = False, **kwargs
    ) -> str:
        df = self.to_df(**df_kwargs)
        kwargs.setdefault("headers", df.columns)
        ret = tabulate(df, **kwargs)

        match truncate:
            case True:
                truncate = get_terminal_size().columns
            case False | int():
                pass
            case "auto":
                from sys import stdout
                if stdout.isatty():
                    truncate = get_terminal_size().columns
                else:
                    truncate = False
            case _:
                raise RuntimeError(f"Invalid {truncate=} value")

        if truncate:
            return trunc(ret, width=truncate)

        return ret

    def to_latex(
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
        data = self.to_dict(**kwargs)
        include = ("value", "central", "sigma", "sigma_rel_perc")
        odict = {
            ".".join(k): v for k, v in data.walkitems() if (k and k[-1] in include)
        }
        logger.log(INFO1, f"Write: {filename}")
        datax(filename, **odict)

    def to_root(self, filename: str) -> None:
        from .export.to_root import ExportToRootVisitor

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
    def update_current(cls, storage: NestedMKDict, *, strict: bool = True):
        if (common_storage := cls.current()) is None:
            return
        if strict:
            common_storage ^= storage
        else:
            common_storage |= storage


_context_storage: list[NodeStorage] = []


class PlotVisitor(NestedMKDictVisitor):
    __slots__ = (
        "_show_all",
        "_folder",
        "_format",
        "_args",
        "_kwargs",
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
        overlay_priority: Sequence[Sequence[str]] = ((),),
        **kwargs,
    ):
        self._show_all = show_all
        self._folder = folder
        self._format = format
        self._args = args
        self._kwargs = kwargs
        self._overlay_priority = tuple(OrderedSet(sq) for sq in overlay_priority)
        self._currently_active_overlay = None
        self._close_on_exitdict = False
        self._active_figures = {}

        if self._show_all:
            self._kwargs["show"] = False
            self._kwargs["close"] = False
        elif self._folder is not None:
            self._kwargs["close"] = False

    def _try_start_join(
        self, key: TupleKey
    ) -> tuple[tuple[str] | None, str | None, bool]:
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
    ) -> tuple[Axes, tuple[str] | None, str | None, bool]:
        from matplotlib.pyplot import sca, subplots

        def mkfig(storekey: TupleKey | None = None) -> Axes:
            fig, ax = subplots(1, 1)
            if storekey is not None:
                self._active_figures[tuple(storekey)] = fig
            return ax

        if force_new:
            return mkfig(), key, None, True

        index_major, index_minor, need_new_figure = self._try_start_join(key)
        if (
            need_new_figure
            or (fig := self._active_figures.get(tuple(index_major))) is None
        ):
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

        if (
            self._currently_active_overlay
            and not self._currently_active_overlay.intersection(key)
        ):
            self._close_figures()
            self._currently_active_overlay = None

    def visit(self, key, output):
        from .plot import plot_auto

        if not isinstance(output, Output) or not output.labels.plottable:
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


class ParametersVisitor(NestedMKDictVisitor):
    __slots__ = ("_kwargs", "_data_list", "_localdata", "_path")
    _kwargs: dict
    _data_list: list[dict]
    _data_dict: NestedMKDict
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
    def data_dict(self) -> NestedMKDict:
        return self._data_dict

    def start(self, dct):
        self._data_list = []
        self._data_dict = NestedMKDict({}, sep=".")
        self._path = ()
        self._paths = []
        self._localdata = []

    def enterdict(self, k, v):
        self._store()
        self._path = k
        self._paths.append(self._path)
        self._localdata = []

    def visit(self, key, value):
        try:
            dct = value.to_dict(**self._kwargs)
        except (AttributeError, IndexError):
            if isinstance(value, ndarray):
                dct = {"shape": value.shape, "label": "data"}
            else:
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
                "shape": len(self._localdata),
                "label": "group",
            }
        )
        self._data_list.extend(self._localdata)
        self._localdata = []

    def stop(self, dct):
        pass
