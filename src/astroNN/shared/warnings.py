import functools
import inspect
import warnings


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if inspect.isclass(func):
            warnings.warn(
                f"The use of class {func.__name__}() is deprecated. It will be removed in future",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Call to function {func.__name__}() is deprecated. It will be removed in future",
                category=DeprecationWarning,
                stacklevel=2,
            )
        return func(*args, **kwargs)

    return new_func


def deprecated_copy_signature(signature_source):
    """
    This is a decorator which can be used to mark old functions
    as deprecated, copy signature from new functions (signature_source).
    It will result in a warning being emitted when the function is used.
    """

    def deco(target):
        @functools.wraps(target)
        def tgt(*args, **kwargs):
            warnings.warn(
                f"Call to function {target.__name__}() is deprecated and will be removed in "
                + f"future. Use {signature_source.__name__}() instead.",
                stacklevel=2,
            )
            inspect.signature(signature_source).bind(*args, **kwargs)
            return target(*args, **kwargs)

        tgt.__signature__ = inspect.signature(signature_source)
        return tgt

    return deco
