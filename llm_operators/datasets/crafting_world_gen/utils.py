import re


def underline_to_pascal(s: str) -> str:
    """Converts a string from underlined to pascal case.

    Example:
        >>> underline_to_pascal("hello_world")
        "HelloWorld"

    Args:
        s: the string to convert.

    Returns:
        the converted string.
    """
    return ''.join([w.capitalize() for w in s.split('_')])


def underline_to_space(s: str) -> str:
    """Converts a string from underlined to a space-separated and lower case string.

    Example:
        >>> underline_to_space("Hello_world")
        "hello world"
    """
    return ' '.join([w.lower() for w in s.split('_')])


def pascal_to_underline(s: str) -> str:
    """Converts a string from pascal to underlined case.

    Example:
        >>> pascal_to_underline("HelloWorld")
        "hello_world"
    """
    return '_'.join([w.lower() for w in re.findall('[A-Z][a-z]*', s)])

