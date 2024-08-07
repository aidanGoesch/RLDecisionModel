import string
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
        """Function that computes the log likelihood of choosing each deck of cards
                RETURNS: n_log_likelihood : list[list], Q_td : list[list], rpe_td : list[list], pc : list[list]"""
        choice_trials = next(x for x in self.trial_rec[0] if x.choice > -1 and x.type == 0)

        alpha = params[0]
        beta = params[1]
        beta_c = params[2]

        reset_trial = self.flags.reset_Q
        if self.flags.reset_Q.isnumeric():
            reset_trial = self.flags.reset_Q

        Q = np.array([np.array([0 for _ in range(NUM_BANDITS)], dtype=float) for _ in range(MAX_TRIALS)])
        pc = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)
        rpe = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)
        run_Q = np.array([0 for i in range(NUM_BANDITS)], dtype=float)