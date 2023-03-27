from dagflow.bundles.load_parameters import load_parameters
from multikeydict.nestedmkdict import NestedMKDict
from pathlib import Path

from pandas import DataFrame
class ParametersWrapper(NestedMKDict):
	def to_dict(self, **kwargs) -> list:
		data = []
		for k, v in self.walkitems():
			k = '.'.join(k)
			dct = v.to_dict(**kwargs)
			dct['path'] = k
			data.append(dct)

		return data

	def to_df(self, **kwargs) -> DataFrame:
		dct = self.to_dict(**kwargs)
		columns = ('path', 'value', 'label')
		df = DataFrame(dct, columns=columns)
		return df

	def to_latex(self) -> str:
		df = self.to_df(label_from='latex')
		return df.to_latex(escape=False)

def model_dayabay_v0():
	storage = ParametersWrapper({}, sep='.')
	datasource = Path('data/dayabay-v0')
	storage |= load_parameters({'path': 'ibd', 'load': datasource/'parameters/pdg2012.yaml'})
	storage |= load_parameters({'path': 'detector', 'load': datasource/'parameters/detector_nprotons_correction.yaml'})
	storage |= load_parameters({'path': 'reactor', 'load': datasource/'parameters/reactor_thermal_power_nominal.yaml'})
	storage |= load_parameters({'path': 'reactor', 'load': datasource/'parameters/detector_eres.yaml'})

	from pprint import pprint
	pprint(storage.object, sort_dicts=False)

	df = storage['constants'].to_df()
	print(df)

	tex = storage['constants'].to_latex()
	print(tex)
