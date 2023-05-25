from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.visitor import NestedMKDictVisitor
from dagflow.output import Output

from typing import Union, Tuple, List, Optional

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
    def read_paths(self) -> None:
        for key, value in self.walkitems():
            labels = getattr(value, 'labels', None)
            if labels is None:
                continue

            key = '.'.join(key)
            labels.paths.append(key)

    def plot(
        self,
        *args,
        **kwargs
    ) -> None:
        from dagflow.plot import plot_auto
        for _, value in self.walkitems():
            if not isinstance(value, Output):
                continue

            plot_auto(value, *args, **kwargs)

    def to_list(self, **kwargs) -> list:
        return self.visit(ParametersVisitor(kwargs)).data_list

    def to_dict(self, **kwargs) -> NestedMKDict:
        return self.visit(ParametersVisitor(kwargs)).data_dict

    def to_df(self, *, columns: Optional[List[str]]=None, **kwargs) -> DataFrame:
        dct = self.to_list(**kwargs)
        if columns is None:
            columns = ['path', 'value', 'central', 'sigma', 'flags', 'label']
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

        if (df['flags']=='').all():
            del df['flags']

        return df

    def to_string(self, **kwargs) -> str:
        df = self.to_df()
        return df.to_string(**kwargs)

    def to_table(
        self,
        *,
        df_kwargs: dict={},
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

        if return_df:
            return tex, df

        return tex

    def to_datax(self, filename: str, **kwargs) -> None:
        data = self.to_dict(**kwargs)
        skip = {'path', 'label', 'flags'} # TODO, add LaTeX label
        odict = {'.'.join(k): v for k, v in data.walkitems() if not (k and k[-1] in skip)}
        datax(filename, **odict)

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
        except AttributeError:
            return
        except IndexError:
            return

        subkey = key[len(self._path):]
        subkeystr = '.'.join(subkey)

        if self._path:
            dct['path'] = f'.. {subkeystr}'
        else:
            dct['path'] = subkeystr

        self._localdata.append(dct)

        self._data_dict[key]=dct

    def exitdict(self, k, v):
        if self._localdata:
            self._data_list.append({
                'path': f"group: {'.'.join(self._path)} [{len(self._localdata)}]"
                })
            self._data_list.extend(self._localdata)
            self._localdata = []
        if self._paths:
            del self._paths[-1]

            self._path = self._paths[-1] if self._paths else ()

    def stop(self, dct):
        pass
