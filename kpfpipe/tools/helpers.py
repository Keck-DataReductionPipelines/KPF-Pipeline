import functools
import traceback

from keckdrpframework.models.arguments import Arguments

def catch_exceptions(func):
    """Wrapper for the _perform method of Pipeline primitives that serves as a try/except
    and allows for a graceful continue in the case of an exception."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Capture the full traceback as a string
            error_msg = traceback.format_exc()
            print(f"An exception occurred: {e}")
            # Optionally, you might want to log the traceback somewhere instead of printing
            return Arguments(name=func.__name__, error=error_msg)  # Return the traceback string to the caller
    return wrapper
