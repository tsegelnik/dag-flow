from dagflow.bundles.load_parameters import load_parameters
from multikeydict.nestedmkdict import NestedMKDict
from pathlib import Path

from typing import Union, Tuple, List, Optional
from pandas import DataFrame

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Sum import Sum

from gindex import GNIndex

class ParametersWrapper(NestedMKDict):
    def to_dict(self, **kwargs) -> list:
        data = []
        for k, v in self.walkitems():
            k = '.'.join(k)
            try:
                dct = v.to_dict(**kwargs)
            except AttributeError:
                continue

            dct['path'] = k
            data.append(dct)

        return data

    def to_df(self, *, columns: Optional[List[str]]=None, **kwargs) -> DataFrame:
        dct = self.to_dict(**kwargs)
        if columns is None:
            columns = ['path', 'value', 'central', 'sigma', 'label']
        df = DataFrame(dct, columns=columns)
        return df

    def to_string(self, **kwargs) -> DataFrame:
        df = self.to_df()
        return df.to_string(**kwargs)

    def to_latex(self, *, return_df: bool=False, **kwargs) -> Union[str, Tuple[str, DataFrame]]:
        df = self.to_df(label_from='latex', **kwargs)
        tex = df.to_latex(escape=False)

        if return_df:
            return tex, df

        return tex

def model_dayabay_v0():
    storage = ParametersWrapper({}, sep='.')
    datasource = Path('data/dayabay-v0')

    index = GNIndex.from_dict({
		('d', 'detector'): ('AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34'),
		('r', 'reactor'): ('DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4'),
		('b', 'background'): ('acc', 'lihe', 'fastn', 'amc', 'alphan'),
		})

    with Graph(close=True) as g:
        storage ^= load_parameters({'path': 'ibd'      , 'load': datasource/'parameters/pdg2012.yaml'})
        storage ^= load_parameters({'path': 'detector' , 'load': datasource/'parameters/detector_nprotons_correction.yaml'})
        storage ^= load_parameters({'path': 'reactor'  , 'load': datasource/'parameters/reactor_thermal_power_nominal.yaml'})
        storage ^= load_parameters({'path': 'eres'     , 'load': datasource/'parameters/detector_eres.yaml'})

        nuisanceall = Sum('nuisance total')
        storage['stat.nuisance.all'] = nuisanceall

        (output for output in storage['stat.nuisance_parts'].values()) >> nuisanceall

    storage['parameter.normalized.eres.eres.b_stat'].value = 1
    storage['parameter.normalized.eres.eres.a_nonuniform'].value = 2

    print('Everything')
    print(storage.to_df())

    print('Parameters')
    print(storage['parameter'].to_df())

    print('Parameters (latex)')
    print(storage['parameter'].to_latex())

    print('Constants (latex)')
    tex = storage['parameter.constant'].to_latex(columns=['path', 'value', 'label'])
    print(tex)

    savegraph(g, "output/dayabay_v0.dot", show='all')
