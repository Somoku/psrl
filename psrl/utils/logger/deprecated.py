import functools
import warnings

def deprecated(reason=None):
    """
    Decorator to mark functions or methods as deprecated.
    Emits a DeprecationWarning when the decorated function/method is called.
    """
    def decorator(func):
        # Build the warning message
        message = f"Call to deprecated {func.__name__}."
        if reason:
            message += f" Reason: {reason}."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    # Allow use without parentheses: @deprecated
    if callable(reason):
        func = reason
        reason = None
        return decorator(func)

    return decorator