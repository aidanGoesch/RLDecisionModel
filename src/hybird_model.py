import numpy as np
from scipy.optimize import minimize

from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params
from src.model import Model

NUM_PARAMS = 5
PARAMS = ["alpha", "beta", "beta_c", "alpha", "beta"]

NUM_BANDITS = 3
MAX_TRIALS = 180

# CONSTANT TEST PARAMS
# alpha_smp = 0.0975
# beta_smp = 5.570
# beta_c = 0.2813
# alpha_td = 0.9575
# beta_td = 19.2978

class HybridModel(Model):
    def __init__(self, subj_idx: int, precomputed_data: dict, trial_rec : dict,
                 verbose : bool = False, very_verbose : bool = False):
        super().__init__("hybrid")

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

        self.flags["resetQ"] = False
        self.flags["num_samples"] = 1

        self.flags["choice_rec"] = precomputed_data["choice_rec"]
        self.flags["combs"] = precomputed_data["combs"]

        super().__dict__["flags"] = self.flags

        self.verbose = verbose
        self.very_verbose = very_verbose

    def likelihood(self, params):
        """Function that computes the log likelihood of choosing each deck of cards
        RETURNS: n_log_likelihood : list[list], Q_td : list[list], rpe_td : list[list], pc : list[list]"""
        num_samples = self.flags["num_samples"]

        # initialize choice trials
        choice_trials = np.array([(x["choice"] > -1 and x["type"] == 0) for x in self.trial_rec[:MAX_TRIALS]])
        choice_trials = np.where(choice_trials)

        alpha_smp = params[0]
        beta_smp = params[1]
        beta_c = params[2]
        alpha_td = params[3]
        beta_td = params[4]

        combs = self.flags["combs"]
        choice_rec = self.flags["choice_rec"]

        Q_td = np.array([np.array([0 for x in range(NUM_BANDITS)], dtype=float) for _ in range(MAX_TRIALS)])
        rpe_td = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)
        run_Q = np.array([0 for _ in range(NUM_BANDITS)], dtype=float)
        pc = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)

        rwdval = [0 for _ in range(NUM_BANDITS)]
        pval = [0 for _ in range(NUM_BANDITS)]

        for i in range(MAX_TRIALS):
            chosen_bandit = self.trial_rec[i]["choice"] + 1
            reward = self.trial_rec[i]["rwdval"]

            if i > 0:
                prev_chosen_bandit = self.trial_rec[i - 1]["choice"] + 1
            else:
                prev_chosen_bandit = -1

            if chosen_bandit == 0:  # Invalid trial. skip
                continue

            for b in range(NUM_BANDITS):
                if not isinstance(combs[i][b], int):  # emulate matlab reshape func behavior - make compatible with int
                    transposed_arr = combs[i][b].T
                    transposed_arr = transposed_arr - 1
                    b_prev_idxs = transposed_arr.reshape(1, -1)
                else:
                    b_prev_idxs = combs[i][b] - 1

                rwdval[b] = choice_rec[b_prev_idxs, 1].T.tolist()
                pval[b] = np.array([alpha_smp * ((1 - alpha_smp) ** (i - b_prev_idxs))])

                if not isinstance(rwdval[b], int) and len(rwdval[b]) < 1:
                    rwdval[b] = np.array([0])
                    pval[b] = np.array([1])

                pval[b] = pval[b] / np.sum(pval[b])

                if isinstance(rwdval[b], list):
                    rwdval[b] = sign(rwdval[b])
                else:
                    rwdval[b] = [sign(rwdval[b])]

            Q_td[i] = run_Q   # save the record of Q values used to make the TD-model based choice
            rpe_td[i] = reward - run_Q[chosen_bandit - 1]
            run_Q[chosen_bandit - 1] += alpha_td * rpe_td[i]

            # find all indices that are not being chosen
            non_chosen_bandits = np.where(np.array(range(1, NUM_BANDITS + 1)) != chosen_bandit)[0]

            # make behavior the same as matlab - add 1 to index
            other_bandit_1 = non_chosen_bandits[0] + 1
            other_bandit_2 = non_chosen_bandits[1] + 1

            I1 = int(other_bandit_1 == prev_chosen_bandit)
            I2 = int(other_bandit_2 == prev_chosen_bandit)
            Ic = int(chosen_bandit == prev_chosen_bandit)

            def compute_rvmat1(x, bandit, I):
                term1 = beta_c * (I - Ic)
                term2 = beta_td * (run_Q[chosen_bandit - 1] - run_Q[bandit])
                term3 = beta_smp * (x - np.array(rwdval[bandit]).flatten())
                return np.exp(term1 - term2 - term3)

            rvmat1 = [compute_rvmat1(x, other_bandit_1 - 1, I1) for x in rwdval[chosen_bandit - 1]]
            rvmat2 = [compute_rvmat1(x, other_bandit_2 - 1, I2) for x in rwdval[chosen_bandit - 1]]

            try:
                rvmat1 = np.concatenate(rvmat1)
                rvmat2 = np.concatenate(rvmat2)
            except ValueError:
                pass

            j = len(rwdval[chosen_bandit - 1])
            k = len(rwdval[other_bandit_1 - 1])
            l = len(rwdval[other_bandit_2 - 1])

            def compute_rvmat(x):
                slice1 = rvmat1[x * k:(x + 1) * k]
                slice2 = rvmat2[x * l:(x + 1) * l]
                reshaped_sum = np.add(slice1, np.vstack(slice2)).reshape(1, -1)
                return reshaped_sum

            rvmat = np.array([compute_rvmat(x) for x in range(j)]).flatten()

            pmat1 = np.array([x * np.array(pval[other_bandit_2 - 1]).T for x in np.array(pval[other_bandit_1 - 1])], dtype=float)
            pmat2 = np.array([x * pmat1.T for x in pval[chosen_bandit - 1]], dtype=float).T.flatten()

            softmax_term = 1. / (1. + rvmat)

            pc[i] = max(np.sum(pmat2 * softmax_term.T), 0.000000000000000000000000001)

        n_log_likelihood = -sum(np.log(pc[choice_trials]))

        n_log_likelihood -= np.log(self.flags["pp_alpha"](alpha_smp))
        n_log_likelihood -= np.log(self.flags["pp_beta"](beta_smp))
        n_log_likelihood -= np.log(self.flags["pp_beta_c"](beta_c))
        n_log_likelihood -= np.log(self.flags["pp_alpha"](alpha_td))
        n_log_likelihood -= np.log(self.flags["pp_beta"](alpha_td))

        if self.verbose:
            print("iteration likelihood:", n_log_likelihood)

        return n_log_likelihood, Q_td, rpe_td, pc
