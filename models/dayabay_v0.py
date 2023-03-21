from dagflow.bundles.load_variables import load_variables
from pathlib import Path

from pprint import pprint

def model_dayabay_v0():
	datasource = Path('data/dayabay-v0')
	vars = load_variables({'load': str(datasource/'parameters/pdg2012.yaml')})

	pprint(vars.object)
