# NOTE: consider to run pytest with "-s" flag to see outputs from this tests

from test_helpers import graph_0

from dagflow.tools.profiling import (
    CountCallsProfiler,
    FrameworkProfiler,
    MemoryProfiler,
    NodeProfiler,
)


def test_minimal_example_g0():
    _, nodes = graph_0()

    node_profiler = NodeProfiler(nodes)
    report = node_profiler.estimate_target_nodes().print_report()
    report.to_csv('output/test_node_report.csv')

    framework_profiler = FrameworkProfiler(nodes)
    report = framework_profiler.estimate_framework_time().print_report()
    report.to_json('output/test_framework_report.json')

    calls_profiler = CountCallsProfiler(nodes)
    calls_profiler.estimate_calls()
    report = calls_profiler.make_report(aggregations=['single', 'std'])

    memory_profiler = MemoryProfiler(nodes)
    report = memory_profiler.estimate_target_nodes().print_report()


def test_full_guide_g0():
    graph, _ = graph_0()

    # Obtain nodes from graph instance
    nodes = graph._nodes


    ## NodeProfiler used for estimates the execution time for each node `n_runs` times.
    #  more precisely, it counts the time of the `Node.function`
    node_profiler = NodeProfiler(nodes, n_runs=1_000)

    # You can specify `sources` and `sinks` to automatically find target nodes,
    #  i.e. nodes that lie on all paths between `sources` and `sinks`.
    # this works for all profiler classes
    _ = NodeProfiler(sources=(nodes[0], nodes[1]), sinks=(nodes[-1],))

    # estimate all target nodes and keep estimations
    node_profiler.estimate_target_nodes()

    # Now we can make reports.
    # Remember, making reports will not work if there are no estimation results

    # 'single' is basically alias for the 'mean' execution time of a single node
    report = node_profiler.make_report(aggregations=["single", "std"])

    # of course you can use 'mean' instead of 'single'
    report = node_profiler.make_report(aggregations=["mean"])

    # also, you can chain these methods
    report = node_profiler.estimate_target_nodes().make_report()

    # You probably want to print results into terminal, consider using `print_report`
    # It returns the same report as `make_report` but also prints it
    report = node_profiler.print_report()

    # For example you want to see estimations for each node, without grouping:
    report = node_profiler.print_report(group_by=None, sort_by="name")

    # Note that default sorting is performed by 'time'
    #  or by first column of aggregation function

    # Saving results is pretty easy, because it's just a pandas.DataFrame
    report.to_csv("output/test_report.csv")
    report.to_json("output/test_report.json")

    # Sometimes it could be helpful to directly estimate one node
    node = nodes[5]
    t = node_profiler.estimate_node(node, n_runs=5_000)
    print(f"{node.name}, time: {t}")


    ## FrameworkProfiler - estimate the overhead of the framework
    # In most cases, you want to measure the overhead for the _entire_ graph.
    framework_profiler = FrameworkProfiler(nodes, n_runs=10_000)

    framework_profiler.estimate_framework_time()

    # Working with the `print_report` and `make_report` methods
    #  is similar for all profiling classes.
    framework_profiler.make_report()
    framework_profiler.print_report()


    ## CountCallsProfiler - obtain the number of calls for nodes
    # May be useful to see how many times each node
    #  was executed during the fit of the model.
    calls_profiler = CountCallsProfiler(nodes)

    # just an example of one-liner:
    # 1. estimate
    # 2. make report
    # 3. print report
    # 4. save to some format
    calls_profiler.estimate_calls().print_report().to_csv('output/test_report.csv')

    calls_profiler.print_report(group_by=None, sort_by='calls', rows=10)


    ## MemoryProfiler - estimate bytes of memory for Inputs/Outputs of given nodes
    memory_profiler = MemoryProfiler(nodes)

    # It is possible to not touch nodes before estimations. Default touch=True
    memory_profiler.estimate_target_nodes(touch=False)

    memory_profiler.print_report()
    memory_profiler.print_report(group_by=None, sort_by='size')
