from os import chdir, getcwd, mkdir
from os.path import isdir

from pytest import fixture, skip


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.

    Automatic change path to the `dag-flow/test` and create `test/output` dir
    """
    path = getcwd()
    lastdir = path.split("/")[-1]
    if lastdir == "dag-flow":  # rootdir
        chdir("./test")
    elif lastdir in (
        "dagflow",
        "example",
        "doc",
        "docs",
        "source",
        "sources",
    ):  # childdir
        chdir("../test")
    if not isdir("output"):
        mkdir("output")


def pytest_addoption(parser):
    parser.addoption("--debug_graph", action="store_true", default=False)


@fixture(scope="session")
def debug_graph(request):
    return request.config.option.debug_graph
