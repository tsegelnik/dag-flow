from os import getcwd, chdir


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    path = getcwd()
    if path.split("/")[-1] != "test":
        chdir("./test")
