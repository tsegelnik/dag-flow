# The DAGFlow package
## Summary

[![python](https://img.shields.io/badge/python-3.10-purple.svg)](https://www.python.org/)
[![pipeline](https://git.jinr.ru/dag-computing/dag-flow/badges/master/pipeline.svg)](https://git.jinr.ru/dag-computing/dag-flow/commits/master)
[![coverage report](https://git.jinr.ru/dag-computing/dag-flow/badges/master/coverage.svg)](https://git.jinr.ru/dag-computing/dag-flow/-/commits/master)
<!--- Uncomment here after adding docs!
[![pages](https://img.shields.io/badge/pages-link-white.svg)](http://dag-computing.pages.jinr.ru/dag-flow)
-->

The **DAGFlow** software is a python implementation of the dataflow programming with the lazy graph evaluation approach.

Main goals:
*  Lazy evaluated directed acyclic graph;
*  Concise connection syntax;
*  Plotting with graphviz;
*  Flexibility. The goal of DAGFlow is not to be efficient, but rather flexible.

The framework is intented to be used for the statistical analysis of the data of *JUNO* and *Daya Bay* neutrino oscillation experiments.

## Installation

### For users (*recommended*)

For regular use, it's best to install [the latest version of the project that's available on PyPi](https://pypi.org/project/dagmodelling/):
```bash
pip install dagmodelling
```

### For developers

We recommend that developers install the package locally in editable mode:
```bash
git clone https://github.com/dagflow-team/dag-flow.git
cd dag-flow
pip install -e .
```
This way, the system will track all the changes made to the source files. This means that developers won't need to reinstall the package or set environment variables, even when a branch is changed.

## Example

For example, let's consider a sum of three input nodes and then a product of the result with another array.

```python
from numpy import arange

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.arithmetic import Sum, Product

# Define a source data
array = arange(3, dtype="d")

# Check predefined Array, Sum and Product
with Graph(debug=debug) as graph:
    # Define nodes
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    m = Product("product")

    # Connect nodes
    (in1, in2, in3) >> s
    (in4, s) >> m
    graph.close()

    print("Result:", m.outputs["result"].data) # must print [0. 3. 12.]
    savegraph(graph, "dagflow_example_1a.png")
```
The printed result must be `[0. 3. 12.]`, and the created image looks as
![](https://raw.githubusercontent.com/dagflow-team/dag-flow/refs/heads/0.9.0/example/dagflow_example_1a.png)


For more examples see [example/example.py](https://github.com/dagflow-team/dag-flow/blob/master/example/example.py) or [tests](https://github.com/dagflow-team/dag-flow/tree/master/tests).

## Additional modules

- Supplementary python modules:
    * [dagflow-reactornueosc](https://git.jinr.ru/dag-computing/dagflow-reactorenueosc) — nodes related to reactor neutrino oscillations
    * [dagflow-detector](https://git.jinr.ru/dag-computing/dagflow-detector) — nodes for the detector responce modelling
    * [dagflow-statistics](https://git.jinr.ru/dag-computing/dagflow-statistics) — statistical analysis and MC
- [Daya Bay model](https://git.jinr.ru/dag-computing/dayabay-model) — test implementation of the Daya Bay oscillation analysis

