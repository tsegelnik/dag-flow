from dictwrapper.dictwrapper import DictWrapper
from storage.storage import Storage

from schema import Schema, Or, Optional, Use, And

IsNumber = Or(float, int, error='Invalid number "{}", expect int of float')
IsNumberOrTuple = Or(IsNumber, (IsNumber,))
IsValuesDict = Schema({str: IsNumberOrTuple})
IsLabel = Or({ 'text': str,
				Optional('latex'): str
			  }, And(str, Use(lambda s: {'text': s})))
IsLabelsDict = Schema({str: IsLabel})
def IsModeFn(mode):
	if mode=='value' or mode==('value',):
		return True
	if len(mode)!=3:
		return False
	if mode[:2]!=('value', 'central'):
		return False
	if mode[2] not in ('sigma_absolute', 'sigma_relative', 'sigma_percent'):
		return False
	return True
IsMode = Schema(IsModeFn, error='Invalid variable mode "{}".')

cfg_schema = Schema({
	'variables': IsValuesDict,
	'labels': IsLabelsDict,
	'mode': IsMode
	})

cfg1 = {
		'variables': {
			'var1': 1.0,
			'var2': 1.0
			},
		'labels': {
			'var1': {
				'text': 'text label 1',
				'latex': r'\LaTeX label 1',
				},
			'var2': 'simple label 2'
			},
		'mode': ('value',)
		}

cfg2 = {
		'variables': {
			'var1': 1.0,
			'var2': 1.0
			},
		'labels': {
			'var1': {
				'text': 'text label 1',
				'latex': r'\LaTeX label 1',
				},
			'var2': 'simple label 2'
			},
		'mode': ('value', 'central', 'sigma_absolute')
		}

cfg3 = {
		'variables': {
			'var1': 1.0,
			'var2': 1.0
			},
		'labels': {
			'var1': {
				'text': 'text label 1',
				'latex': r'\LaTeX label 1',
				},
			'var2': 'simple label 2'
			},
		'mode': ('value', 'central', 'sigma_relative')
		}

cfg4 = {
		'variables': {
			'var1': 1.0,
			'var2': 1.0
			},
		'labels': {
			'var1': {
				'text': 'text label 1',
				'latex': r'\LaTeX label 1',
				},
			'var2': 'simple label 2'
			},
		'mode': ('value', 'central', 'sigma_percent')
		}

def test_schema():
	cfg_schema.validate(cfg1)
	cfg_schema.validate(cfg2)
	cfg_schema.validate(cfg3)
	cfg_schema.validate(cfg4)

def test_load_variables():
	pass
