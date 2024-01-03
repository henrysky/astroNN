def nn_obj_lookup(identifier, module_obj=None, module_name="default_obj"):
    """
    Lookup astroNN.nn function by name

    :param identifier: identifier
    :type identifier: str
    :param module_obj: globals()
    :type module_obj: Union([Nonetype, dir])
    :param module_name: module english name
    :type module_name: str
    :return: Looked up function
    :rtype: function
    :History: 2018-Apr-28 - Written - Henry Leung (University of Toronto)
    """
    function_name = identifier
    fn = module_obj.get(function_name)
    if fn is None:
        raise ValueError("Unknown function: " + module_name + "." + function_name)
    return fn
