"""
Miscelaneous common utility functions.
"""


def censor_string(string: str, slice_size: int = 6, fill_str: str = " ... ") -> str:
    """
    Censors the middle of a string if to save space. Effectively shows the
    first and last `slice_size` characters and `fill_str` in the middle. Only
    censors if it would save space.
    """

    max_lentgh = 2 * slice_size + len(fill_str)
    return (
        string
        if len(string) <= max_lentgh
        else f"{string[:slice_size]}{fill_str}{string[-slice_size:]}"
    )
