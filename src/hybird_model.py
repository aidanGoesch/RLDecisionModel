import numpy as np

from math import inf
from types import SimpleNamespace
from scipy.optimize import minimize


from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params

NUM_PARAMS = 5
PARAMS = ["alpha", "beta", "beta_c", "alpha", "beta"]

NUM_BANDITS = 3
MAX_TRIALS = 180

# Flags for ctx_hybrid = pp_alpha, pp_beta, and pp_betaC
# Params = alpha, beta, beta_c, alpha, beta

class Model:
    def __init__(self, subj_idx: int, input_data: dict, precomputed_data: SimpleNamespace, trial_rec : SimpleNamespace):
        self.input_data = input_data
        self.precomputed = precomputed_data
        self.trial_rec = trial_rec
        self.subj_idx = subj_idx
        self.flags = FLAGS    # load defaults

        self.flags["resetQ"] = False
        self.flags["num_samples"] = 1

        self.flags["choice_rec"] = precomputed_data.choice_rec
        self.flags["combs"] = precomputed_data.combs

        self.verbose = False
        self.very_verbose = False

        self.results = {
            "num_params": NUM_PARAMS,
            "n_log_lik": inf,
            "file": None   # I don't think we need this for now
        }

    def likelihood(self, alpha, beta, beta_c):
        """Function that computes the log likelihood of choosing each deck of cards"""
        num_samples = self.flags.num_samples

        # initialize choice trials
        choice_trials = np.array([(x.choice > -1 and x.type == 0) for x in self.trial_rec[:MAX_TRIALS]])   # change this later
        choice_trials = np.argmax(choice_trials, axis=1)

        # Change these later
        alpha_smp = alpha
        beta_smp = beta
        beta_c = beta_c
        alpha_td = alpha
        beta_td = beta

        combs = self.flags.combs[num_samples]
        choice_rec = self.flags.choice_rec

        Q_td = np.array([np.array([0 for x in range(NUM_BANDITS)]) for y in range(MAX_TRIALS)])
        rpe_td = np.array([0 for i in range(MAX_TRIALS)])
        run_Q = np.array([0 for i in range(NUM_BANDITS)])
        pc = np.array([0 for i in range(MAX_TRIALS)])

        rwdval = np.array([])
        pval = {}

        for i in range(MAX_TRIALS):
            chosen_bandit = self.trial_rec[i].choice + 1
            reward = self.trial_rec[i].rwdval

            if i > 0:
                prev_chosen_bandit = self.trial_rec[i - 1].choice + 1
            else:
                prev_chosen_bandit = -1

            if chosen_bandit == 0:  # Invalid trial. skip
                continue

            for b in range(NUM_BANDITS):
                transposed_arr = combs[i][b].T
                b_prev_idxs = transposed_arr.reshape(1, -1)

                rwdval[b] = choice_rec[b_prev_idxs, 1].T.tolist()
                pval[b] = np.array([alpha_smp * ((1 - alpha_smp) ** (i - b_prev_idxs))])

                if len(rwdval[b]) < 1:
                    rwdval[b] = [0]
                    pval[b] = [1]

                pval[b] = pval[b] / np.sum(pval[b])
                rwdval[b] = np.array(list(map(sign, rwdval[b])))

            Q_td[i] = run_Q   # save the record of Q values used to make the TD-model based choice
            rpe_td[i] = reward - run_Q[chosen_bandit]
            run_Q[chosen_bandit] = run_Q[chosen_bandit] + alpha_td * rpe_td[i]

            non_chosen_bandits = np.where(range(1, NUM_BANDITS + 1) != chosen_bandit)  # find all of the indices that are not being chosen

            other_bandit_1 = non_chosen_bandits[0]
            other_bandit_2 = non_chosen_bandits[1]

            I1 = other_bandit_1 == prev_chosen_bandit
            I1 = other_bandit_2 == prev_chosen_bandit
            Ic = chosen_bandit == prev_chosen_bandit

            def compute_value(x, bandit):
                term1 = beta_c * (I1 - Ic)
                term2 = beta_td * (run_Q[chosen_bandit] - run_Q[bandit])
                term3 = beta_smp * (x - rwdval[bandit])
                return np.exp(term1 - term2 - term3)

            rvmat1 = np.array([compute_value(x, other_bandit_1) for x in rwdval[chosen_bandit]])
            rvmat2 = np.array([compute_value(x, other_bandit_2) for x in rwdval[chosen_bandit]])

            rvmat1 = np.concatenate(rvmat1)
            rvmat2 = np.concatenate(rvmat2)

            j = len(rwdval[chosen_bandit])
            k = len(rwdval[non_chosen_bandits[0]])
            l = len(rwdval[non_chosen_bandits[1]])

            def compute_rvmat(x):
                slice1 = rvmat1[x * k:(x + 1) * k]
                slice2 = rvmat2[x * l:(x + 1) * l]
                reshaped_sum = (slice1 + slice2.T).reshape(1, -1)
                return reshaped_sum

            rvmat = np.array([compute_rvmat(x) for x in range(j)])
            rvmat = np.vstack(rvmat)

            pmat1 = np.array([x * pval[other_bandit_2] for x in pval[other_bandit_1]])
            pmat2 = np.array([x * pmat1.T for x in pval[chosen_bandit]])

            softmax_term = 1 / (1 + rvmat)
            pc[i] = max(sum(pmat2 * softmax_term), 0)    # this might cause problems -- change to 1e-32

        n_log_likelihood = -sum(np.log(pc[choice_trials]))

        n_log_likelihood -= np.log(self.flags.pp_alpha[alpha_smp])
        n_log_likelihood -= np.log(self.flags.pp_beta[beta_smp])
        n_log_likelihood -= np.log(self.flags.pp_beta_C[beta_c])
        n_log_likelihood -= np.log(self.flags.pp_alpha[alpha_td])
        n_log_likelihood -= np.log(self.flags.pp_beta[alpha_td])

        return n_log_likelihood


    def fit_model(self, subj_idx):
        print(f"---- FITTING SUBJECT {self.subj_idx}----")

        start, n_unchanged_trials = 0, 0

        while n_unchanged_trials < ITERATIONS:
            start += 1

            # pick random starting values for the params
            x_0 = [np.random.uniform(PARAM_CONSTANTS[PARAMS[x]][0]) for x in range(NUM_PARAMS)]

            transformed_x_0 = transform_params(x_0, PARAMS)

            options = {"disp": False}

            result = minimize(self.likelihood, transformed_x_0, options)

            transformed_xf = transform_params(result.x, PARAMS)


            if self.verbose:
                print("DEBUG")

            if result.status != 1:  # this might be wrong
                print("Failed to converge")
                continue
            elif self.very_verbose:
                print("DEBUG")

            if start == 1 or result.fun  < self.results["n_log_lik"]:
                if self.verbose:
                    print("DEBUG")

                n_unchanged_trials = 0   # reset to zero if nLogLik decreases

                self.results["n_log_lik"] = result.fun
                self.results["params"] = result.x
                self.results["transformed_params"] = transformed_xf
                self.results["model"] = "Hybrid"
                self.results["exit flag"] = result.status
                self.results["output"] = result.message

                _, Q, rpe, pc = self.likelihood(*result.x)

                self.results["run_Q"] = Q
                self.results["pc"] = pc
                self.results["rpe"] = rpe

                use_log_log = result.fun

                # cleaner way to write this
                if not np.isinf(np.log(self.flags["pp_alpha"][result.x[0]])) and not np.isnan(np.log(self.flags["pp_alpha"][result.x[0]])):
                    use_log_log += np.log(self.flags["pp_alpha"][result.x[0]])

                if not np.isinf(np.log(self.flags["pp_beta"][result.x[1]])) and not np.isnan(np.log(self.flags["pp_beta"][result.x[1]])):
                    use_log_log += np.log(self.flags["pp_beta"][result.x[1]])

                if not np.isinf(np.log(self.flags["pp_beta_c"][result.x[2]])) and not np.isnan(np.log(self.flags["pp_beta_c"][result.x[2]])):
                    use_log_log += np.log(self.flags["pp_beta_c"][result.x[2]])

                if not np.isinf(np.log(self.flags["pp_alpha"][result.x[3]])) and not np.isnan(np.log(self.flags["pp_alpha"][result.x[3]])):
                    use_log_log += np.log(self.flags["pp_alpha"][result.x[3]])

                if not np.isinf(np.log(self.flags["pp_beta"][result.x[4]])) and not np.isnan(np.log(self.flags["pp_beta"][result.x[4]])):
                    use_log_log += np.log(self.flags["pp_alpha"][result.x[4]])

                self.results["use_log_log"] = use_log_log
                self.results["AIC"] = 2 * len(result.x) + 2 * use_log_log
                self.results["BIC"] = 0.5 * len(result.x) * np.log(180) + use_log_log

            else:
                n_unchanged_trials += 1

        self.display_results()


    def display_results(self):
        """Method that displays the results of the fitting procedure"""
        return None if self.results["n_log_lik"] is inf  # makes sure that it is printing something

        print(f"---- RESULTS----")
        for key, value in self.results:
            print(f"{key}: {value}")


