from os import getcwd, chdir


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
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
