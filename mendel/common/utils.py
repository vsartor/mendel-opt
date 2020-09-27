"""
Miscelaneous common utility functions.
"""


def censor_string(string: str) -> str:
    _SLICE_SIZE = 6
    _FILL_STR = " ... "
    _MAX_LENGTH = 2 * _SLICE_SIZE + len(_FILL_STR)

    return string if len(string) <= _MAX_LENGTH else f"{string[:_SLICE_SIZE]}{_FILL_STR}{string[-_SLICE_SIZE:]}"
