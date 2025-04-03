# TODO: reorder imports
from pandas import DataFrame
from dagflow.tools.profiling import FitSimulationProfiler

from test_helpers import graph_0, graph_1


def test_estimate_separately():
    _, nodes = graph_0()

    # see graph structure in test/output/test_profiling_graph_0.png
    a0, a1, a2, _, _, _, _, _, _, s3, _, mdvdt = nodes
    params = [a0, a1, a2]
    endpoints = [s3, mdvdt]

    fit_profiling = FitSimulationProfiler(
        mode="parameter-wise",
        parameters=params,
        endpoints=endpoints,
        n_runs=10_000
    )

    fit_profiling.estimate_fit()
    report = fit_profiling.make_report()

    assert isinstance(report, DataFrame)
    assert not report.empty

    report = fit_profiling.print_report()

    assert isinstance(report, DataFrame)
    assert not report.empty


