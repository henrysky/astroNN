def lamost_default_dr(dr=None):
    """
    Check if dr argument is provided, if none then use default

    :param dr: data release
    :type dr: Union(int, NoneType)
    :return: data release
    :rtype: int
    :History: 2018-May-13 - Written - Henry Leung (University of Toronto)
    """
    if dr is None:
        dr = 5
        print(f'dr is not provided, using default dr={5}')
    else:
        raise ValueError('Only LAMOST DR5 is supported')

    return dr
