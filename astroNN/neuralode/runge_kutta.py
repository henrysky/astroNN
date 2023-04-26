import tensorflow as tf


###################################################################
# 4th order RK ####################################################
###################################################################
@tf.function
def RK4Step(func, y, t, h, k1, *args, **kwargs):
    k2 = func(y + 0.5 * h * k1, t + 0.5 * h, *args, **kwargs)
    k3 = func(y + 0.5 * h * k2, t + 0.5 * h, *args, **kwargs)
    k4 = func(y + h * k3, t + h, *args, **kwargs)
    return (k1 + 2.0 * (k2 + k3) + k4) / 6.0


@tf.function
def RK4TwoStep(func, y, t, h, k1, *args, **kwargs):
    step1 = RK4Step(func, y, t, 0.5 * h, k1)
    t1 = t + 0.5 * h
    y1 = y + 0.5 * h * step1
    k1 = func(y1, t1, *args, **kwargs)
    step2 = RK4Step(func, y1, t1, 0.5 * h, k1)
    return (step1 + step2) / 2.0


@tf.function
def rk4_core(n, func, x, t, hmax, h, tol, uround, *args, **kwargs):
    # array to store the result
    result = tf.TensorArray(dtype=x.dtype, size=t.shape[0])
    result = result.write(0, x)

    # last and current point of the numerical integration
    ycurr = ylast = qcurr = qlast = x
    tcurr = tlast = t[0]
    fcurr = func(ycurr, tcurr, *args, **kwargs)
    totalerr = tf.constant(0.0, dtype=x.dtype)
    totalvar = tf.constant(0.0, dtype=x.dtype)
    i = tf.constant(0, dtype=tf.int32)

    for _t in t[1:]:
        # remember that t == t[i+1], result goes to yout[i+1]
        while tf.greater((_t - tcurr) * h, 0):
            # advance the integration
            k1 = RK4Step(func, ycurr, tcurr, h, fcurr, *args, **kwargs)
            k2 = RK4TwoStep(func, ycurr, tcurr, h, fcurr, *args, **kwargs)

            scale = tf.reduce_max(tf.abs(k2))
            steperr = tf.reduce_max(tf.abs(k1 - k2)) / 2.0
            # compute the ideal step size factor and sanitize the result to prevent ridiculous changes
            hfac = (tol * scale / (uround + steperr)) ** 0.25
            hfac = tf.reduce_min([10.0, tf.reduce_max([0.01, hfac])])

            # repeat the step if there is a significant step size correction
            if tf.logical_and(
                tf.less(tf.abs(h * hfac), hmax),
                tf.logical_or(tf.less(hfac, 0.6), tf.greater(hfac, 3.0)),
            ):
                # recompute with new step size
                h = h * hfac
                k2 = RK4TwoStep(func, ycurr, tcurr, h, fcurr, *args, **kwargs)

            # update and cycle the integration points
            ylast = tf.identity(ycurr)
            ycurr = ycurr + h * k2
            tlast = tf.identity(tcurr)
            tcurr = tcurr + h
            flast = tf.identity(fcurr)
            fcurr = func(ycurr, tcurr, *args, **kwargs)
            # cubic Bezier control points
            qlast = ylast + (tcurr - tlast) / 3.0 * flast
            qcurr = ycurr - (tcurr - tlast) / 3.0 * fcurr

            totalvar = totalvar + h * scale
            totalerr = (1.0 + h * scale) * totalerr + h * steperr

        # now tlast <= t <= tcurr, can interpolate the value for yout[i+1] using the cubic Bezier formula
        s = (_t - tlast) / (tcurr - tlast)
        temp_result = (1 - s) ** 2.0 * (
            (1 - s) * ylast + 3.0 * s * qlast
        ) + s**2.0 * (3.0 * (1.0 - s) * qcurr + s * ycurr)
        # temp_result = tf.ones_like(x)*tcurr
        i = i + 1
        result = result.write(i, temp_result)

    return result.stack()


def rk4(func=None, x=None, t=None, tol=None, tf_float=tf.float32, *args, **kwargs):
    if tf_float == tf.float32:
        tf_float = tf.float32
        uround = tf.constant(1.1920929e-07, tf_float)
        if tol is None:
            tol = tf.constant(1e-6, tf_float)
    elif tf_float == tf.float64:
        tf_float = tf.float64
        uround = tf.constant(2.220446049250313e-16, tf_float)
        if tol is None:
            tol = tf.constant(1e-10, tf_float)
    else:
        raise TypeError(f"Data type {tf_float} not supported")

    x = tf.cast(x, tf_float)
    t = tf.cast(t, tf_float)
    # initialization
    n = x.shape[0]

    h = t[1] - t[0]
    hmax = tf.abs(t[-1] - t[0])

    result = rk4_core(n, func, x, t, hmax, h, tol, uround, *args, **kwargs)

    if "aux" in kwargs.keys():
        return result, t, kwargs["aux"]
    else:
        return result, t
