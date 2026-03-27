import numpy as np
import warnings

def validate_array(arr, context="array", response="warn", check_finite=True, check_positive=True):
    """
    Validate that an array contains finite and/or positive values.

    Parameters
    ----------
    arr : array-like
        Input array to validate.
    context : str
        Descriptive name for messages.
    response : {'warn', 'error', 'silent'}
        How to report issues:
        - 'warn'  : issue warnings for all problems
        - 'error' : raise ValueError listing all problems
        - 'silent': ignore all problems
    check_finite : bool, default=True
        Whether to check for NaN or infinite values.
    check_positive : bool, default=True
        Whether to check for negative values.
    """
    if response not in ("warn", "error", "silent"):
        raise ValueError("response must be 'warn', 'error', or 'silent'")

    issues = []

    if check_finite:
        nan_count = np.sum(np.isnan(arr))
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values detected in {context}")

        inf_count = np.sum(np.isinf(arr))
        if inf_count > 0:
            issues.append(f"{inf_count} Non-finite values detected in {context}")

    if check_positive:
        neg_count = np.sum(arr < 0)
        if neg_count > 0:
            issues.append(f"{neg_count} Negative values detected in {context}")

    if not issues:
        return

    msg = "; ".join(issues)

    if response == "warn":
        warnings.warn(msg)
    elif response == "error":
        raise ValueError(msg)
    # silent -> do nothing