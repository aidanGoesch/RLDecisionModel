import math

from scipy.stats import beta, norm


# Experiment constants
PARAM_CONSTANTS = {
    "alpha": [[0, 1], 0.05],
    "beta": [[0, 20], 0.5],
    "beta_c": [[-3, 3], 0.1],
}

# Flags that are used for all models
FLAGS = { "pp_alpha": lambda x: beta.pdf(x, 1.1, 1.1) }

low_bound, up_bound = PARAM_CONSTANTS["beta"][0]
FLAGS["pp_beta"] = lambda x: norm.pdf((x - low_bound) / (up_bound - low_bound), 0, 10)

low_bound, up_bound = PARAM_CONSTANTS["beta_c"][0]
FLAGS["pp_betaC"] = lambda x: norm.pdf((x - low_bound) / (up_bound - low_bound), 0, 10)


# helper function
def sign(x: int):
    return math.copysign(1, x) if x != 0 else 0