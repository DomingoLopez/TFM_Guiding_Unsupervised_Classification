import warnings
import functools

def deprecated(message=None):
    """
    Decorator to mark functions as deprecated
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn_message = f"{func.__name__} is deprecated and should not be used."
            if message:
                warn_message += f" {message}"
            warnings.warn(warn_message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    
    # Allow @deprecated and @deprecated("mensaje")
    if callable(message):
        return decorator(message)
    return decorator