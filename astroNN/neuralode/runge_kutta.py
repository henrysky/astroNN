import numpy as np
import tensorflow as tf


###################################################################
# 4th order RK ####################################################
###################################################################
@tf.function
def RK4Step(func, y, t, h, k1):
    k2 = func(y + 0.5 * h * k1, t + 0.5 * h)
    k3 = func(y + 0.5 * h * k2, t + 0.5 * h)
    k4 = func(y + h * k3, t + h)
    return (k1 + 2. * (k2 + k3) + k4) / 6.0


@tf.function
def RK4TwoStep(func, y, t, h, k1):
    step1 = RK4Step(func, y, t, 0.5 * h, k1)
    t1 = t + 0.5 * h
    y1 = y + 0.5 * h * step1
    k1 = func(y1, t1)
    step2 = RK4Step(func, y1, t1, 0.5 * h, k1)
    return (step1 + step2) / 2.


@tf.function
def rk4_core(n, func, x, t, hmax, h, tol, tf_float, uround):
    # array to store the result
    result = tf.TensorArray(dtype=tf_float, size=t.shape[0])
    result = result.write(0, x)

    # last and current point of the numerical integration
    ycurr = ylast = qcurr = qlast = x
    tcurr = tlast = t[0]
    fcurr = func(ycurr, tcurr)
    totalerr = tf.constant(0., dtype=tf_float)
    totalvar = tf.constant(0., dtype=tf_float)
    i = tf.constant(0, dtype=tf.int32)

    for _t in t[1:]:
        # remember that t == t[i+1], result goes to yout[i+1]
        while tf.greater((_t - tcurr) * h, 0):
            # advance the integration
            k1 = RK4Step(func, ycurr, tcurr, h, fcurr)
            k2 = RK4TwoStep(func, ycurr, tcurr, h, fcurr)

            scale = tf.reduce_max(tf.abs(k2))
            steperr = tf.reduce_max(tf.abs(k1 - k2)) / 2.
            # compute the ideal step size factor and sanitize the result to prevent ridiculous changes
            hfac = (tol * scale / (uround + steperr)) ** 0.25
            hfac = tf.reduce_min([10., tf.reduce_max([0.01, hfac])])

            # repeat the step if there is a significant step size correction
            if tf.logical_and(tf.less(tf.abs(h * hfac), hmax),
                                      tf.logical_or(tf.greater(tf.constant(0.6, dtype=tf_float), hfac), tf.greater(hfac, 3.))):
                # recompute with new step size
                h = h * hfac
                k2 = RK4TwoStep(func, ycurr, tcurr, h, fcurr)

            # update and cycle the integration points
            ylast = tf.identity(ycurr)
            ycurr = ycurr + h * k2
            tlast = tf.identity(tcurr)
            tcurr = tcurr + h
            flast = tf.identity(fcurr)
            fcurr = func(ycurr, tcurr)
            # cubic Bezier control points
            qlast = ylast + (tcurr - tlast) / 3. * flast
            qcurr = ycurr - (tcurr - tlast) / 3. * fcurr

            totalvar = totalvar + h * scale
            totalerr = (1. + h * scale) * totalerr + h * steperr

        # now tlast <= t <= tcurr, can interpolate the value for yout[i+1] using the cubic Bezier formula
        s = (_t - tlast) / (tcurr - tlast)
        temp_result = (1 - s) ** 2. * ((1 - s) * ylast + 3. * s * qlast) + s ** 2. * (3. * (1. - s) * qcurr + s * ycurr)
        # temp_result = tf.ones_like(x)*tcurr
        result = result.write(i + 1, temp_result)
        i = i+1

    # map back to Tensor
    stack_components = lambda x: x.stack()
    result = tf.nest.map_structure(stack_components, result)

    return result


def rk4(func=None, x=None, t=None, tol=None, precision=tf.float32, args=()):
    if precision == tf.float32:
        tf_float = tf.float32
        np_float = np.float32
    elif precision == tf.float64:
        tf_float = tf.float64
        np_float = np.float64
    else:
        raise TypeError(f"Data type {precision} not understood")

    # machine limit related info from numpy
    unsigned_int_max = tf.constant(np.iinfo(np.int64).max)
    uround = tf.constant(np.finfo(np_float).eps)

    if tol is None:
        tol = 1e-5 if tf_float == tf.float32 else 1e-10

    x = tf.cast(x, tf_float)
    t = tf.cast(t, tf_float)
    # initialization
    n = x.shape[0]

    h = t[1] - t[0]
    hmax = tf.abs(t[-1] - t[0])

    result = rk4_core(n, func, x, t, hmax, h, tol, tf_float, uround)

    return result, t
