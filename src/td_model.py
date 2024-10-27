import numpy as np

from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params
from src.model import Model

NUM_PARAMS = 3
PARAMS = ["alpha", "beta", "beta_c"]

NUM_BANDITS = 3
MAX_TRIALS = 180

class TDModel(Model):
    def __init__(self, subj_idx: int, precomputed_data: dict, trial_rec: dict,
                 verbose: bool = False, very_verbose: bool = False):
        super().__init__("td")

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

        # hard coded from fit_model.m
        self.flags["reset_Q"] = False
        self.flags["signed_LR"] = False

    def likelihood(self, params):
        """Function that computes the log likelihood of choosing each deck of cards
                RETURNS: n_log_likelihood : list[list], Q : list[list], rpe : list[list], pc : list[list]"""
        choice_trials = np.array([(x["choice"] > -1 and x["type"] == 0) for x in self.trial_rec[:MAX_TRIALS]])
        choice_trials = np.where(choice_trials)

        alpha = params[0]
        beta = params[1]
        beta_c = params[2]

        reset_trial = self.flags["reset_Q"]

        Q = np.array([np.array([0 for _ in range(NUM_BANDITS)], dtype=float) for _ in range(MAX_TRIALS)])
        pc = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)
        rpe = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)
        run_Q = np.array([0 for _ in range(NUM_BANDITS)], dtype=float)

        for trial_idx in range(MAX_TRIALS):
            Q[trial_idx, :] = run_Q

            chosen_bandit = self.trial_rec[trial_idx]["choice"] + 1
            reward = self.trial_rec[trial_idx]["rwdval"]

            if chosen_bandit == 0: continue  # Invalid trial. Skip it.

            if trial_idx > 0:
                prev_chosen_bandit = self.trial_rec[trial_idx - 1]["choice"] + 1
            else:
                prev_chosen_bandit = -1

            non_chosen_bandits = np.where(np.array(range(1, NUM_BANDITS + 1)) != chosen_bandit)[0]

            # make behavior the same as matlab - add 1 to index
            other_bandit_1 = non_chosen_bandits[0] + 1
            other_bandit_2 = non_chosen_bandits[1] + 1

            I1 = int(other_bandit_1 == prev_chosen_bandit)
            I2 = int(other_bandit_2 == prev_chosen_bandit)
            Ic = int(chosen_bandit == prev_chosen_bandit)

            term1 = np.exp(beta_c * (I1 - Ic) + beta * (run_Q[other_bandit_1 - 1] - run_Q[chosen_bandit - 1]))
            term2 = np.exp(beta_c * (I2 - Ic) + beta * (run_Q[other_bandit_2 - 1] - run_Q[chosen_bandit - 1]))

            pc[trial_idx] = 1 / (1 + term1 + term2)

            rpe[trial_idx] = reward - run_Q[chosen_bandit - 1]

            run_Q[chosen_bandit - 1] += alpha * rpe[trial_idx]

        n_log_likelihood = -sum(np.log(pc[choice_trials]))

        n_log_likelihood -= np.log(self.flags["pp_alpha"](alpha))
        n_log_likelihood -= np.log(self.flags["pp_beta"](beta))
        n_log_likelihood -= np.log(self.flags["pp_beta_c"](beta_c))

        if self.verbose:
            print("iteration likelihood:", n_log_likelihood)

        return n_log_likelihood, Q, rpe, pc

