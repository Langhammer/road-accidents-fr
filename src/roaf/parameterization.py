"""This module provides functions to parameterize the execution of notebooks."""
import numpy as np


def reduce_to_one_parameter_combination(parameter_grid, reduction_factor=1):
    """Select first parameter combination of given parameter grid.

    When using GridSearchCV, the execution time increases with the number of parameters and
    dimensions. This function selects only the first combination. It also returns the number of
    combinations in the parameter grid that was given.
    """

    reduction_factor *= np.prod(
        [len(this_params) for this_params in parameter_grid.values()]
    )
    params = [[this_params[0]] for this_params in parameter_grid.values()]
    parameter_grid = dict(zip(parameter_grid.keys(), params))
    return parameter_grid, reduction_factor


def set_parameter(parameter, std_value, fast_value, fast_execution, reduction_factor=1):
    """Set a parameter with the standard or fast value if not already specified.

    Sets the parameter with the standard value in standard execution,
    the fast value in fast execution or leave it as it is if it is already specified.
    """
    if parameter is None:
        if fast_execution:
            parameter = fast_value
        else:
            parameter = std_value

    reduction_factor = parameter / std_value
    return parameter, reduction_factor
