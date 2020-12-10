import io
import os
from pathlib import Path


def parse(line, delim=',', quote='\"', esc='\\'):
    """
    Parses a single CSV row.

    :param line
        The string representing the CSV line to parse. This is usually
        a line in a CSV file obtained via `f.readline()` or of the likes.
    
    :param delim
        The cell delimiter. By default this is the standard comma.
    
    :param quote
        The quote character used to encase complex cell information that
        may otherwise break the entire CSV structure, for example, by
        containing an illegal delimiter character.
    
    :param esc
        The escape character used to escape sensitive characters.
    
    :return
        A list of parsed cells.

    If `line` is None, this function does nothing and returns None.
    
    If any of the `delim`, `quote`, `esc`, variables are None, a 
    ValueError.

    If the length of any `delim`, `quote`, `esc` is not a single 
    character, this routine also throws a ValueError.
    """
    if line is None:
        return None

    if delim is None or \
       quote is None or \
       esc   is None:
        raise ValueError("delim, quote, and/or esc cannot be None")
    
    if len(delim) != 1 and \
       len(quote) != 1 and \
       len(esc)   != 1:
        raise ValueError("len of delim, quote, and esc must be 1")

    cells = []
    buf = io.StringIO()

    in_quote = False
    esc_next = False
    for c in line:
        if c == '\n' and not in_quote:
            break

        if esc_next:
            buf.write(c)
            esc_next = False
            continue

        if c == esc:
            esc_next = True
            continue

        if c == quote:
            in_quote = not in_quote
            continue

        if c == delim and not in_quote:
            cells.append(buf.getvalue())
            buf = io.StringIO()
            continue

        buf.write(c)
    
    leftover = buf.getvalue()
    if len(leftover) > 0:
        cells.append(leftover)
    
    return cells


def render(*args, delim=',', quote='\"', esc='\\', enquote=True):
    """
    Renders a CSV row given the cells in `args`.

    :param args
        The list of cells to format into a CSV row ready for
        output to an external medium. If args is a list of
        a single value, being a list, then that list will
        be used as the cells for the row.
    
    :param delim
        The cell delimiter. By default this is the standard comma.
    
    :param quote
        The quote character used to encase complex cell information that
        may otherwise break the entire CSV structure, for example, by
        containing an illegal delimiter character.
    
    :param esc
        The escape character used to escape sensitive characters.
    
    :param enquote
        Whether to automatically surround cells with quotes. If False,
        heuristics will determine whether the cell must be enquoted.

    :return 
        The string representing the formatted CSV row.

    If `delim` is None or `quote` is None or `esc` is None, this function throws
    a ValueError.

    If len(`delim`) != 1 or len(`quote`) != 1 or len(`esc`) != 1, this function
    """
    if delim is None or \
       quote is None or \
       esc   is None:
        raise ValueError("delim, quote, and/or esc cannot be None")
    
    if len(delim) != 1 and \
       len(quote) != 1 and \
       len(esc) != 1:
        raise ValueError("len of delim, quote, and esc must be 1")

    if len(args) == 1 and isinstance(args[0], Iterable):
        return render(*args[0])

    def _format_cell(data):
        nonlocal delim, quote, esc, enquote
        formatted = str(data).replace(quote, esc + quote)
        if enquote or delim in formatted:
            return quote + formatted + quote
        return formatted

    return delim.join([_format_cell(x) for x in args])


def write(*args, stream, delim=',', quote='\"', esc='\\', enquote=True):
    """
    Writes cell formatted information to the specified stream.

    :param stream
        The stream object to which to write the formatted CSV row to.
    
    If `stream` is None, a ValueError is raised.
    """
    if stream is None:
        raise ValueError("stream must not be None")

    stream.write(f"{render(*args, delim=delim, quote=quote, esc=esc, enquote=enquote)}\n")
