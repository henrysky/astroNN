import functools
import inspect
import warnings


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        if inspect.isclass(func):
            warnings.warn(f"The use of class {func.__name__}() is deprecated. It will be removed in future",
                          category=DeprecationWarning, stacklevel=2)
        else:
            warnings.warn(f"Call to function {func.__name__}() is deprecated. It will be removed in future",
                          category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
