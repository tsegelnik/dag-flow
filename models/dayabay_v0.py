from dagflow.bundles.load_parameters import load_parameters
from multikeydict.nestedmkdict import NestedMKDict
from pathlib import Path

from typing import Union, Tuple, List, Optional
from pandas import DataFrame

from dagflow.graph import Graph
from dagflow.graphviz import savegraph

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
			columns = ['path', 'value', 'central', 'sigma', 'normvalue', 'label']
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

	with Graph() as g:
		storage |= load_parameters({'path': 'ibd'      , 'load': datasource/'parameters/pdg2012.yaml'})
		storage |= load_parameters({'path': 'detector' , 'load': datasource/'parameters/detector_nprotons_correction.yaml'})
		storage |= load_parameters({'path': 'reactor'  , 'load': datasource/'parameters/reactor_thermal_power_nominal.yaml'})
		storage |= load_parameters({'path': 'eres'     , 'load': datasource/'parameters/detector_eres.yaml'})

	# from pprint import pprint
	# pprint(storage.object, sort_dicts=False)

	print('Everything')
	print(storage.to_df())

	print('Parameters')
	print(storage['parameter'].to_df())

	print('Parameters (latex)')
	print(storage['parameter'].to_latex())

	print('Constants (latex)')
	tex = storage['parameter.constant'].to_latex(columns=['path', 'value', 'label'])
	print(tex)

	savegraph(g, "output/dayabay_v0.dot")
