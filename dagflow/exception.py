class CriticalError(Exception):
    pass


class NoncriticalError(Exception):
    pass


def UnconnectedInput(cls, name):
    return NoncriticalError(
        f"UnconnectedInput: {type(cls).__name__} {cls.name}: "
        f"The following input is not connected: '{name}'!"
    )
