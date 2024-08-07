import numpy as np

from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params
from src.model import Model


NUM_PARAMS = 3
PARAMS = ["alpha", "beta", "beta_p"]

NUM_BANDITS = 3
MAX_TRIALS = 180

class TDModel(Model):
    def __init__(self, subj_idx: int, precomputed_data: dict, trial_rec: dict,
                 verbose: bool = False, very_verbose: bool = False):
        super().__init__()

        super().__dict__["params"] = PARAMS
        super().__dict__["num_params"] = NUM_PARAMS

        super().__dict__["verbose"] = verbose
        super().__dict__["very_verbose"] = very_verbose

        super().__dict__["results"] = {
            "num_params": NUM_PARAMS,
            "n_log_lik": np.inf,
        }

        self.subj_idx = subj_idx
        self.trial_rec = trial_rec

        # load defaults
        self.flags = FLAGS

        self.num_samples = self.trial_rec["num_samples"]

    def likelihood(self, params):
        pass

