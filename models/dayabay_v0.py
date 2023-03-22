from dagflow.bundles.load_variables import load_variables
from pathlib import Path

from pprint import pprint

def model_dayabay_v0():
	datasource = Path('data/dayabay-v0')
	vars1 = load_variables({'path': 'ibd', 'load': datasource/'parameters/pdg2012.yaml'})
	vars2 = load_variables({'path': 'reactor', 'load': datasource/'parameters/reactor_e_per_fission.yaml'})

	pprint(vars1.object)
	pprint(vars2.object)
