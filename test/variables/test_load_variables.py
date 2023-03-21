from dagflow.graphviz import savegraph
from dagflow.graph import Graph
from dagflow.bundles.load_variables import load_variables

cfg1 = {
        'variables': {
            'var1': 1.0,
            'var2': 1.0,
            'sub1': {
                'var3': 2.0
                }
            },
        'format': 'value',
        'state': 'variable',
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-1'
                },
            'var2': 'simple label 2',
            },
        }
cfg1a = {
        'variables': {
            'var1': 1.0,
            'var2': 1.0,
            'sub1': {
                'var3': 2.0
                }
            },
        'format': 'value',
        'state': 'fixed',
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-1'
                },
            'var2': 'simple label 2',
            },
        }

cfg2 = {
        'variables': {
            'var1': (1.0, 1.0, 0.1),
            'var2': (1.0, 2.0, 0.1),
            'sub1': {
                'var3': (2.0, 1.0, 0.1)
                }
            },
        'path': 'sub.folder',
        'format': ('value', 'central', 'sigma_absolute'),
        'state': 'fixed',
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-2'
                },
            'var2': 'simple label 2'
            },
        }

cfg3 = {
        'variables': {
            'var1': (1.0, 1.0, 0.1),
            'var2': (1.0, 2.0, 0.1),
            'sub1': {
                'var3': (2.0, 3.0, 0.1)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                'name': 'v1-3'
                },
            'var2': 'simple label 2'
            },
        'format': ('value', 'central', 'sigma_relative'),
        'state': 'fixed',
        }

cfg4 = {
        'variables': {
            'var1': (1.0, 1.0, 10),
            'var2': (1.0, 2.0, 10),
            'sub1': {
                'var3': (2.0, 3.0, 10)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        'format': ('value', 'central', 'sigma_percent'),
        'state': 'variable',
        }

cfg5 = {
        'variables': {
            'var1': (1.0, 10),
            'var2': (2.0, 10),
            'sub1': {
                'var3': (3.0, 10)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        'format': ('central', 'sigma_percent'),
        'state': 'variable',
        }

from pprint import pprint
def test_load_variables_v01():
    cfgs = (cfg1, cfg1a, cfg2, cfg3, cfg4, cfg5)
    with Graph(close=True) as g:
        for i, cfg in enumerate(cfgs):
            vars = load_variables(cfg)
            print(cfg['state'])
            print(i, end=' ')
            pprint(vars.object)

    savegraph(g, 'output/test_load_variables.pdf', show='all')
