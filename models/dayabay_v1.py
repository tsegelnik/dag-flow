from dagflow.bundles.load_parameters import load_parameters
from pathlib import Path

from pprint import pprint

def model_dayabay_v1():
	datasource = Path('data/dayabay-v1')
	vars = load_parameters({'path': 'ibd', 'load': datasource/'parameters/pdg2020.yaml'})

	pprint(vars.object)
