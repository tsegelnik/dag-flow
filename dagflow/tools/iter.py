from typing import Generator, Sequence, TypeVar, Union

T = TypeVar("T")


def iter_sequence_not_string(
    seq_or_str: Union[str, T, Sequence[T], None]
) -> Generator[Union[T, str], None, None]:
    """Iterate a sequence o yield an object. Strings are not sequences here."""
    if isinstance(seq_or_str, str):
        yield seq_or_str
    elif isinstance(seq_or_str, Sequence):
        yield from seq_or_str
    elif seq_or_str is not None:
        yield seq_or_str
