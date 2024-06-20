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
FLAGS["pp_betaC"] = lambda x: norm.pdf((x - low_bound) / (up_bound - low_bound), 0, 10)


# helper functions
def sign(x: int):
    """implementation of the MatLab sign function"""
    return math.copysign(1, x) if x != 0 else 0



def transform_params(params, param_names):

    transformed_params = params[::]
    for i in range(len(params)):
        param_range = PARAM_CONSTANTS[param_names[i]][0]

        transformed_params[i] = -np.log(-1 + (param_range[1] - param_range[0]) / (params[i]-param_range[0]))

    return transformed_params
