import numpy as np

from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params
from src.model import Model

# alpha - learning rate; beta - softmax temp; beta_p (gets turned into beta_c) - perseveration

NUM_PARAMS = 3
PARAMS = ["alpha", "beta", "beta_c"]

NUM_BANDITS = 3
MAX_TRIALS = 180

# CONSTANT TEST PARAMS
# alpha = 0.957166948242946
# beta = 9.707512974456824
# beta_c = 1.801682813332801

class SamplingModel(Model):
    def __init__(self, subj_idx : int, precomputed_data : dict, trial_rec : dict,
                 verbose : bool = False, very_verbose : bool = False):
        super().__init__("sample")

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

        self.flags["choice_rec"] = precomputed_data["choice_rec"]
        self.flags["combs"] = precomputed_data["combs"]

        self.num_samples = 1  # hard coded from fit_model.m

    def likelihood(self, params):
        """Function that computes the log likelihood of choosing each deck of cards
        RETURNS: n_log_likelihood : list[list], Q : list[list], rpe : list[list], pc : list[list]"""

        # select a valid choice trial
        choice_trials = np.array(
            [(x["choice"] > -1 and x["type"] == 0) for x in self.trial_rec[:MAX_TRIALS]])
        choice_trials = np.where(choice_trials)

        alpha = params[0]
        beta = params[1]
        beta_c = params[2]

        combs = self.flags["combs"]
        choice_rec = self.flags["choice_rec"]

        Q = np.array([np.array([0 for _ in range(NUM_BANDITS)], dtype=float) for _ in range(MAX_TRIALS)])
        pc = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)
        rpe = np.array([0 for _ in range(MAX_TRIALS)], dtype=float)

        rwdval = [0 for _ in range(MAX_TRIALS)]
        pval = [0 for _ in range(NUM_BANDITS)]

        # skip the first trial
        pc[0] = 0.5

        for trial_idx in range(1, MAX_TRIALS):
            chosen_bandit = self.trial_rec[trial_idx]["choice"] + 1
            prev_chosen_bandit = self.trial_rec[trial_idx - 1]["choice"] + 1

            if chosen_bandit == 0: continue  # Invalid trial. Skip it.

            for b in range(NUM_BANDITS):
                # b_prev_idx = np.array(combs[trial_idx][b]).T.reshape(1)  # reshape specific location in combs to be a row vector

                if not isinstance(combs[trial_idx][b], int):  # emulate matlab reshape func behavior - make compatible with int
                    transposed_arr = combs[trial_idx][b].T
                    transposed_arr = transposed_arr - 1
                    b_prev_idx = transposed_arr.reshape(1, -1)
                else:
                    b_prev_idx = combs[trial_idx][b] - 1

                rwdval[b] = choice_rec[b_prev_idx, 1].T.tolist()  # make sure this is an ndarray

                pval[b] = [alpha * ((1 - alpha) ** (trial_idx - np.array(b_prev_idx)))]

                if not isinstance(rwdval[b], int) and len(rwdval[b]) == 0:
                    rwdval[b] = [0]
                    pval[b] = [1]

                pval[b] /= np.sum(pval[b])
                if isinstance(rwdval[b], list):
                    rwdval[b] = sign(rwdval[b])
                else:
                    rwdval[b] = [sign(rwdval[b])]

                Q[trial_idx][b] = np.sum(rwdval[b] * pval[b])
                rpe[trial_idx] = self.trial_rec[trial_idx]["rwdval"] - Q[trial_idx][chosen_bandit - 1]

            non_chosen_bandits = np.where(np.array(range(1, NUM_BANDITS + 1)) != chosen_bandit)[0]

            # make behavior the same as matlab - add 1 to index
            other_bandit_1 = non_chosen_bandits[0] + 1
            other_bandit_2 = non_chosen_bandits[1] + 1

            I1 = int(other_bandit_1 == prev_chosen_bandit)
            I2 = int(other_bandit_2 == prev_chosen_bandit)
            Ic = int(chosen_bandit == prev_chosen_bandit)

            def compute_rvmat1(x, bandit, I):
                term1 = beta_c * (I - Ic)
                term2 = beta * (x - np.array(rwdval[bandit]).flatten()).T

                return np.exp(term1 - term2)

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

            pc[trial_idx] = max(np.sum(pmat2 * softmax_term.T), 0.000000000000000000000000001)

        n_log_likelihood = -sum(np.log(pc[choice_trials]))

        n_log_likelihood -= np.log(self.flags["pp_alpha"](alpha))
        n_log_likelihood -= np.log(self.flags["pp_beta"](beta))
        n_log_likelihood -= np.log(self.flags["pp_beta_c"](beta_c))

        if self.verbose:
            print("iteration likelihood:", n_log_likelihood)

        return n_log_likelihood, Q, rpe, pc


#TODO: Debug this model

if __name__ == "__main__":
    pass
