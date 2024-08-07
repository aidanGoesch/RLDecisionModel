import numpy as np

from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params
from src.model import Model

# alpha - learning rate; beta - softmax temp; beta_p - perseveration

NUM_PARAMS = 3
PARAMS = ["alpha", "beta", "beta_p"]

NUM_BANDITS = 3
MAX_TRIALS = 180

class SamplingModel(Model):
    def __init__(self, subj_idx : int, precomputed_data : dict, trial_rec : dict,
                 verbose : bool = False, very_verbose : bool = False):
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

        # select a valid choice trial
        choice_trials = next(x for x in self.trial_rec[0] if x.choice > -1 and x.type == 0)

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
            chosen_bandit = self.trial_rec[trial_idx].choice + 1
            prev_chosen_bandit = self.trial_rec[trial_idx - 1].choice + 1

            if chosen_bandit == 0: continue  # Invalid trial. Skip it.

            for b in range(NUM_BANDITS):
                b_prev_idx = np.array(combs[trial_idx][b]).T.reshape(1)  # reshape specific location in combs to be a row vector
                rwdval[b] = [choice_rec[b_prev_idx, 1].T]  # make sure this is an ndarray

                if len(rwdval[b]) == 0:
                    rwdval[b] = [0]
                    pval[b] = [1]

                pval[b] /= np.sum(rwdval[b])
                rwdval[b] = sign(rwdval[b])

                Q[trial_idx][b] = rwdval[b] * pval[b]
                rpe[trial_idx] = self.trial_rec[trial_idx].rwdval - Q[trial_idx][chosen_bandit]

            non_chosen_bandits = np.where(np.array(range(1, NUM_BANDITS + 1)) != chosen_bandit)[0]

            # make behavior the same as matlab - add 1 to index
            other_bandit_1 = non_chosen_bandits[0] + 1
            other_bandit_2 = non_chosen_bandits[1] + 1

            I1 = int(other_bandit_1 == prev_chosen_bandit)
            I2 = int(other_bandit_2 == prev_chosen_bandit)
            Ic = int(chosen_bandit == prev_chosen_bandit)

            def compute_rvmat1(x, bandit, I):
                term1 = beta_c * (I - Ic)
                term2 = beta * (x - rwdval[bandit]).T

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


#TODO: Debug this model

if __name__ == "__main__":
    pass
