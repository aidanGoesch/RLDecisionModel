import math

import numpy as np

from scipy.stats import beta, norm


# Experiment constants
PARAM_CONSTANTS = {
    "alpha": [[0, 1], 0.05],
    "beta": [[0, 20], 0.5],
    "beta_c": [[-3, 3], 0.1],
}

ITERATIONS = 5

# Flags that are used for all models
FLAGS = { "pp_alpha": lambda x: beta.pdf(x, 1.1, 1.1) }

low_bound, up_bound = PARAM_CONSTANTS["beta"][0]
FLAGS["pp_beta"] = lambda x: norm.pdf((x - low_bound) / (up_bound - low_bound), 0, 10)

low_bound, up_bound = PARAM_CONSTANTS["beta_c"][0]
FLAGS["pp_beta_c"] = lambda x: norm.pdf((x - low_bound) / (up_bound - low_bound), 0, 10)


# helper functions
def sign(x: int | list):
    """implementation of the MatLab sign function"""
    def foo(i):
        if isinstance(i, list):
            return [sign(i) for i in x]
        else:
            return math.copysign(1, x) if x != 0 else 0

    return foo(x)


def transform_params(params : list, param_names : list, minimizing: bool = False):

    transformed_params = params[::]
    for i in range(len(params)):
        param_range = PARAM_CONSTANTS[param_names[i]][0]
        if minimizing:  # min + [max-min]./[1+exp(-x)]
            transformed_params[i] = param_range[0] + (param_range[1] - param_range[0]) / (1 + np.exp(-params[i]))
        else:
            transformed_params[i] = -np.log(-1 + (param_range[1] - param_range[0]) / (params[i]-param_range[0]))

    return transformed_params




if __name__ == "__main__":
    # print(format_rwd_val([[1], 1, [[1], 1, [1]]]))
    # print(sign([0, 1]))
    # print(list(map(sign, [[0], 1])))
    x = [0.6469239473475671, -1.3404036392300163, 3.3682370745009322, 0.07216152970584128, 0.4192123986941138]
    x = transform_params(x, ["alpha", "beta", "beta_c", "alpha", "beta"], minimizing=True)

    print(x)
    # print([type(i) for i in x])

    print(FLAGS["pp_alpha"](1.4247013627734044))
