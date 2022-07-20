class CriticalError(Exception):
    pass


class NoncriticalError(Exception):
    pass


def UnconnectedInput(name):
    return NoncriticalError(
        f"Warning: The required input is not connected: '{name}'!"
    )
