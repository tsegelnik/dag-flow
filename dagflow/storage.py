from multikeydict.typing import Key
from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.visitor import NestedMKDictVisitor
from .output import Output
from .input import Input
from .node import Node
from .logger import logger, DEBUG

from typing import Union, Tuple, List, Optional, Dict, Mapping

from tabulate import tabulate
from pandas import DataFrame
from LaTeXDatax import datax

from numpy import nan
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 100)

from shutil import get_terminal_size

def trunc(text: str, width: int) -> str:
    return '\n'.join(line[:width] for line in text.split('\n'))

class NodeStorage(NestedMKDict):
    __slots__ = ('_remove_connected_inputs',)
    _remove_connected_inputs: bool

    def __init__(self, *args, remove_connected_inputs: bool=True, **kwargs):
        kwargs.setdefault('sep', '.')
        kwargs.setdefault('recursive_to_others', True)
        super().__init__(*args, **kwargs)

        self._remove_connected_inputs = remove_connected_inputs

    def plot(
        self,
        *args,
        show_all: bool = False,
        **kwargs
    ) -> None:
        from .plot import plot_auto
        from matplotlib.pyplot import subplots, show

        if show_all:
            kwargs['show'] = False
            kwargs['close'] = False
            def mkfigure(): return subplots(1,1)
        else:
            def mkfigure(): pass

        for _, output in self.walkitems():
            if not isinstance(output, Output) or not output.labels.plottable:
                continue

            mkfigure()
            plot_auto(output, *args, **kwargs)

        if show_all:
            show()

    #
    # Connectors
    #
    def __rshift__(self, other: NestedMKDict):
        if not isinstance(other, NestedMKDict):
            raise RuntimeError("Operator >> RHS should be NestedMKDict")

        nconnections = 0
        to_remove = []
        for keyleft, valueleft in self.walkitems():
            if not isinstance(valueleft, Output):
                raise RuntimeError(f"Invalid left value type for {keyleft}: {type(valueleft)} (need Output)")
            setleft = set(keyleft)

            for keyright, valueright in other.walkitems():
                if not isinstance(right, (Input, Node)):
                    raise RuntimeError(f"Invalid right value type for {keyright}: {type(valueright)} (need Input/Node)")
                setright = set(keyright)

                if not setleft.issubset(setright):
                    continue

                valueleft >> valueright

                if self._remove_connected_inputs:
                    to_remove.append(keyright)
                nconnections+=1

        if nconnections==0:
            raise RuntimeError("No connections are done")

        for key in to_remove:
            print('remove', key)
            del other[key]

    #
    # Finalizers
    #
    def read_paths(self) -> None:
        for key, value in self.walkitems():
            labels = getattr(value, 'labels', None)
            if labels is None:
                continue

            key = '.'.join(key)
            labels.paths.append(key)

    # def process_indices(self, index_values: Tuple[str, ...]) -> None:
    #     from multikeydict.flatten import flatten
    #     newdict = flatten(self, index_values)
    #     for k, v in newdict.items():
    #         self[k] = v

    def read_labels(self, source: Union[NestedMKDict, Dict], *, strict: bool=False) -> None:
        source = NestedMKDict(source, sep='.')

        processed_keys = set()
        def get_label(key):
            try:
                # if strict:
                #     labels = source.pop(key, delete_parents=True)
                # else:
                labels = source(key)
            except (KeyError, TypeError):
                pass
            else:
                processed_keys.add(key)
                return labels

            keyleft = list(key[:-1])
            keyright = []
            while keyleft:
                groupkey = keyleft+['group']
                try:
                    labels = source(groupkey)
                except (KeyError, TypeError):
                    keyright.insert(0, keyleft.pop())
                else:
                    processed_keys.add(tuple(groupkey))
                    return labels

        for key, object in self.walkitems():
            if not isinstance(object, (Node, Output)):
                continue

            logger.log(DEBUG, f"Look up label for {'.'.join(key)}")
            if (labels := get_label(key)) is None:
                continue
            logger.log(DEBUG, "... found")

            if isinstance(object, Node):
                object.labels.update(labels)
            elif isinstance(object, Output):
                object.labels = object.labels or {}
                object.labels.update(labels)

        if strict:
            for key in processed_keys:
                source.delete_with_parents(key)
            if source:
                raise RuntimeError(f'The following label groups were not used: {tuple(source.keys())}')

    def remove_connected_inputs(self, key: Key=()):
        source = self(key)
        to_remove = [key for key,input in source.walkitems() if isinstance(input, Input)]
        for key in to_remove:
            source.delete_with_parents(key)

    #
    # Converters
    #

    def to_list(self, **kwargs) -> list:
        return self.visit(ParametersVisitor(kwargs)).data_list

    def to_dict(self, **kwargs) -> NestedMKDict:
        return self.visit(ParametersVisitor(kwargs)).data_dict

    def to_df(self, *, columns: Optional[List[str]]=None, **kwargs) -> DataFrame:
        dct = self.to_list(**kwargs)
        if columns is None:
            columns = ['path', 'value', 'central', 'sigma', 'flags', 'shape', 'label']
        df = DataFrame(dct, columns=columns)

        df.insert(4, 'sigma_rel_perc', df['sigma'])
        df['sigma_rel_perc'] = df['sigma']/df['central']*100.
        df['sigma_rel_perc'].mask(df['central']==0, nan, inplace=True)

        for key in ('central', 'sigma', 'sigma_rel_perc'):
            if df[key].isna().all():
                del df[key]
            else:
                df[key].fillna('-', inplace=True)

        df['value'].fillna('-', inplace=True)
        df['flags'].fillna('', inplace=True)
        df['label'].fillna('', inplace=True)
        df['shape'].fillna('', inplace=True)

        if (df['flags']=='').all():
            del df['flags']

        return df

    def to_str(self, **kwargs) -> str:
        df = self.to_df()
        return df.to_str(**kwargs)

    def to_table(
        self,
        *,
        df_kwargs: Mapping={},
        truncate: Union[int, bool] = False,
        **kwargs
    ) -> str:
        df = self.to_df(**df_kwargs)
        kwargs.setdefault('headers', df.columns)
        ret = tabulate(df, **kwargs)

        if truncate:
            if isinstance(truncate, bool):
                truncate = get_terminal_size().columns

            return trunc(ret, width=truncate)

        return ret

    def to_latex(self, *, return_df: bool=False, **kwargs) -> Union[str, Tuple[str, DataFrame]]:
        df = self.to_df(label_from='latex', **kwargs)
        tex = df.to_latex(escape=False)

        return tex, df if return_df else tex

    def to_datax(self, filename: str, **kwargs) -> None:
        data = self.to_dict(**kwargs)
        include = ('value', 'central', 'sigma', 'sigma_rel_perc')
        odict = {'.'.join(k): v for k, v in data.walkitems() if (k and k[-1] in include)}
        datax(filename, **odict)

    def to_root(self, filename: str) -> None:
        from .export.to_root import ExportToRootVisitor
        visitor = ExportToRootVisitor(filename)
        self.visit(visitor)

    #
    # Current storage, context
    #
    @staticmethod
    def current() -> Optional["NodeStorage"]:
        return _context_storage[-1] if _context_storage else None

    def __enter__(self) -> "NodeStorage":
        _context_storage.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _context_storage.pop()!=self:
            raise RuntimeError("NodeStorage: invalid context exit")

_context_storage: List[NodeStorage] = []

class ParametersVisitor(NestedMKDictVisitor):
    __slots__ = ('_kwargs', '_data_list', '_localdata', '_path')
    _kwargs: dict
    _data_list: List[dict]
    _data_dict: NestedMKDict
    _localdata: List[dict]
    _paths: List[Tuple[str, ...]]
    _path: Tuple[str, ...]
    # _npars: List[int]

    def __init__(self, kwargs: dict):
        self._kwargs = kwargs
        # self._npars = []

    @property
    def data_list(self) -> List[dict]:
        return self._data_list

    @property
    def data_dict(self) -> NestedMKDict:
        return self._data_dict

    def start(self, dct):
        self._data_list = []
        self._data_dict = NestedMKDict({}, sep='.')
        self._path = ()
        self._paths = []
        self._localdata = []

    def enterdict(self, k, v):
        if self._localdata:
            self.exitdict(self._path, None)
        self._path = k
        self._paths.append(self._path)
        self._localdata = []

    def visit(self, key, value):
        try:
            dct = value.to_dict(**self._kwargs)
        except (AttributeError, IndexError):
            return

        if dct is None:
            return

        subkey = key[len(self._path):]
        subkeystr = '.'.join(subkey)

        dct['path'] = self._path and f'.. {subkeystr}' or subkeystr
        self._localdata.append(dct)

        self._data_dict[key]=dct

    def exitdict(self, k, v):
        if self._localdata:
            self._data_list.append({
                'path': f"group: {'.'.join(self._path)} [{len(self._localdata)}]",
                'shape': len(self._localdata),
                'label': 'group'
                })
            self._data_list.extend(self._localdata)
            self._localdata = []
        if self._paths:
            del self._paths[-1]

            self._path = self._paths[-1] if self._paths else ()

    def stop(self, dct):
        pass
