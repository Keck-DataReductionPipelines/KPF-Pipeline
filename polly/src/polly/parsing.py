"""
polly

parsing

Contains functions to parse various types of input, such as dates, filenames, numeric
ranges, etc.
"""

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from polly.kpf import LFC_ORDER_INDICES, ORDERLETS, THORIUM_ORDER_INDICES, TIMESOFDAY


class Mask(NamedTuple):
    """
    A simple class to more easily couple a date, time of day, and orderlet together.
    """

    date: datetime
    timeofday: str
    orderlet: str


def parse_filename(
    filename: Path | str | list[Path] | list[str],
) -> tuple[datetime, str, str]:
    """
    Parse a filename that is supposed to contain a date, time of day, and orderlet, and
    return a tuple of these three elements.

    Args:
        filename (Path | str | list[Path] | list[str]): The input file(name)(s)

    Returns:
        tuple[datetime, str, str]: A tuple containing the date, time of day, and
            orderlet, or a list of such tuples.
    """

    if isinstance(filename, list):
        return [parse_filename(f) for f in filename]

    if isinstance(filename, str):
        filename = Path(filename)

    filename = filename.name
    datestr, timeofday, orderlet, *_ = filename.split("_")[:3]
    date = parse_yyyymmdd(datestr)

    return Mask(date=date, timeofday=timeofday, orderlet=orderlet)


def parse_yyyymmdd(yyyymmdd: str | float) -> datetime:
    """
    Parse both strings and floats (or ints) representing dates in the format "YYYYMMDD"
    or "YYYY-MM-DD" (if a string) and return the corresponding datetime object.

    Args:
        yyyymmdd (str | float): The input string or numeric value representing a date,
            perhaps a command-line argument

    Returns:
        datetime: A datetime object representing the date passed as input
    """

    if yyyymmdd == "now":
        return datetime.now()

    if isinstance(yyyymmdd, float):
        yyyymmdd = str(int(yyyymmdd))
    elif isinstance(yyyymmdd, int):
        yyyymmdd = str(yyyymmdd)

    # Handle dates like "2024-12-31"
    elif isinstance(yyyymmdd, str) and "-" in yyyymmdd:
        yyyymmdd = "".join(yyyymmdd.split("-"))
        # Now it should be "20241231"

    assert isinstance(yyyymmdd, str) and len(yyyymmdd) == 8  # noqa: PLR2004

    yyyy = int(yyyymmdd[0:4])
    mm = int(yyyymmdd[4:6])
    dd = int(yyyymmdd[6:8])

    return datetime(year=yyyy, month=mm, day=dd)


def parse_num_list(string_list: str) -> list[int]:
    """
    Adapted from Julian Stürmer's PyEchelle code

    Converts a string specifying a range of numbers (e.g. '1-3') into a list of those
    numbers ([1,2,3])
    """

    m = re.match(r"(\d+)(?:-(\d+))?$", string_list)
    if not m:
        raise argparse.ArgumentTypeError(
            f"'{string_list}' is not a range or number."
            + "Expected forms like '1-12' or '6'."
        )

    start = m.group(1)
    end = m.group(2) or start

    return list(range(int(start), int(end) + 1))


def parse_orders(orders_str: str) -> list[int]:
    """
    Wrapper around parse_num_list
    """

    if (orders_str == "all") or (orders_str is None):
        return list(range(67))
    if orders_str == "lfc":
        return LFC_ORDER_INDICES
    if orders_str == "thorium":
        return THORIUM_ORDER_INDICES

    try:
        return parse_num_list(orders_str)
    except Exception as e:
        print(f"Exception raised when parsing orders: {e}")
        print("Returning ALL orders")
        return list(range(67))


def parse_timesofday(timesofday: str) -> list:
    """
    Parse a string representing a time of day or a list of times of day, and return a
    list of these times of day.

    Args:
        timesofday (str): The input string

    Returns:
        list: A list of time of day strings (each one of the form "morn", "eve",
        "night", "midnight)
    """

    if (timesofday == "all") or (timesofday is None):
        return TIMESOFDAY

    if "," in timesofday:
        return timesofday.split(sep=",")

    return [timesofday]


def parse_orderlets(orderlets: str) -> list:
    """
    Parse a string representing an orderlet or a list of orderlets, and return a list of
    these orderlets.

    Args:
        orderlets (str): The input string

    Returns:
        list: A list of orderlet strings (each one of the form "SCI1", "SCI2", "SCI3",
        "CAL", "SKY")
    """

    if (orderlets == "all") or (orderlets is None):
        return ORDERLETS

    if "," in orderlets:
        orderlets = orderlets.split(sep=",")
        for ol in orderlets:
            assert ol in ORDERLETS
        return orderlets

    assert orderlets in ORDERLETS
    return [orderlets]


def parse_bool(input_string: str) -> bool:
    """
    Parse a boolean input string and return the corresponding boolean value.

    Args:
        input_string (str): The input string (a command line argument)

    Raises:
        argparse.ArgumentTypeError

    Returns:
        bool: The corresponding boolean value
    """

    if isinstance(input_string, bool):
        return input_string
    if input_string.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    if input_string.lower() in ["no", "false", "f", "n", "0"]:
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_orderlet_name(orderlet: str) -> str:
    """
    A simple helper function to get the non-numeric part of the orderlet name, used to
    build the relevant FITS header keyword to access data.

    eg. for 'SCI1' we need 'GREEN_SCI_FLUX1', so return 'SCI'
    """

    if orderlet.startswith("SCI"):
        return "SCI"

    return orderlet


def get_orderlet_index(orderlet: str) -> str:
    """
    A simple helper function to get only the numeric part of the orderlet name, used to
    build the relevant FITS header keyword to access data.

    eg. for 'SCI1' we need 'GREEN_SCI_FLUX1', so return '1'
    """

    if orderlet.startswith("SCI"):
        return orderlet[-1]

    return ""
