import numpy as np
from scipy.optimize import minimize

from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params

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


class HybridModel:
    def __init__(self, subj_idx: int, precomputed_data: dict, trial_rec : dict,
                 verbose : bool = False, very_verbose : bool = False):
        self.subj_idx = subj_idx
        self.trial_rec = trial_rec

        # load defaults
        self.flags = FLAGS

        self.flags["resetQ"] = False
        self.flags["num_samples"] = 1

        self.flags["choice_rec"] = precomputed_data["choice_rec"]
        self.flags["combs"] = precomputed_data["combs"]

        self.verbose = verbose
        self.very_verbose = very_verbose

        self.results = {
            "num_params": NUM_PARAMS,
            "n_log_lik": np.inf,
        }

        # stops unnecessary error warnings during running
        np.seterr(divide='ignore')

    def likelihood(self, params):
        """Function that computes the log likelihood of choosing each deck of cards
        RETURNS: n_log_likelihood : list[list], Q_td : list[list], rpe_td : list[list], pc : list[list]"""
        num_samples = self.flags["num_samples"]

        # initialize choice trials
        choice_trials = np.array([(x["choice"] > -1 and x["type"] == 0) for x in self.trial_rec[:MAX_TRIALS]])   # change this later
        choice_trials = np.where(choice_trials)

        alpha_smp = params[0]
        beta_smp = params[1]
        beta_c = params[2]
        alpha_td = params[3]
        beta_td = params[4]

        combs = self.flags["combs"]
        choice_rec = self.flags["choice_rec"]

        Q_td = np.array([np.array([0 for x in range(NUM_BANDITS)], dtype=float) for y in range(MAX_TRIALS)])
        rpe_td = np.array([0 for i in range(MAX_TRIALS)], dtype=float)
        run_Q = np.array([0 for i in range(NUM_BANDITS)], dtype=float)
        pc = np.array([0 for i in range(MAX_TRIALS)], dtype=float)

        rwdval = [0 for i in range(NUM_BANDITS)]
        pval = [0 for i in range(NUM_BANDITS)]

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


    def fit(self):
        """Fit the model to the data passed into the constructor"""
        start, n_unchanged_trials = 0, 0

        def f(x):
            """Function to be minimized by scipy.minimize"""
            tmp = (transform_params(x, PARAMS, minimizing=True))

            if self.very_verbose:
                print("x:", x)
                print("t:", tmp)

            return self.likelihood(params=tmp)[0]

        while n_unchanged_trials < ITERATIONS:
            print(f"Trial {n_unchanged_trials + 1} started")
            start += 1

            # pick random starting values for the params
            x_0 = [np.random.uniform(*PARAM_CONSTANTS[PARAMS[x]][0]) for x in range(NUM_PARAMS)]

            # initial parameters
            transformed_x_0 = transform_params(x_0, PARAMS)

            # options for minimization function - gtol determines how precise the minimization is
            options = {"disp": False, 'gtol': 0.5, 'maxiter': 1000}

            result = minimize(fun=f, x0=transformed_x_0, options=options)

            transformed_xf = transform_params(result.x, PARAMS, minimizing=True)


            if self.verbose:
                print(f"valid_x0={x_0} valid_xf={transformed_xf}\nraw_x0={transformed_x_0}  raw_xf={result.x}")

            if not result.success:
                print("Failed to converge")
                print(result.message)
                continue
            elif self.very_verbose:
                print("n_log_likelihood:", result.fun)

            if start == 1 or result.fun < self.results["n_log_lik"]:
                if self.verbose:
                    print(f"old n_log_likelihood: {self.results["n_log_lik"]}")
                    print(f"new n_log_likelihood: {result.fun}")

                n_unchanged_trials = 0   # reset to zero if nLogLik decreases

                # save results into attribute
                self.results["n_log_lik"] = result.fun
                self.results["params"] = result.x
                self.results["transformed_params"] = transformed_xf
                self.results["model"] = "Hybrid"
                self.results["exit flag"] = result.status
                self.results["output"] = result.message

                _, Q, rpe, pc = self.likelihood(params = result.x)

                self.results["run_Q"] = Q
                self.results["pc"] = pc
                self.results["rpe"] = rpe

                use_log_log = result.fun

                if not np.isinf(np.log(self.flags["pp_alpha"](result.x[0]))) and not np.isnan(np.log(self.flags["pp_alpha"](result.x[0]))):
                    use_log_log += np.log(self.flags["pp_alpha"](result.x[0]))

                if not np.isinf(np.log(self.flags["pp_beta"](result.x[1]))) and not np.isnan(np.log(self.flags["pp_beta"](result.x[1]))):
                    use_log_log += np.log(self.flags["pp_beta"](result.x[1]))

                if not np.isinf(np.log(self.flags["pp_beta_c"](result.x[2]))) and not np.isnan(np.log(self.flags["pp_beta_c"](result.x[2]))):
                    use_log_log += np.log(self.flags["pp_beta_c"](result.x[2]))

                if not np.isinf(np.log(self.flags["pp_alpha"](result.x[3]))) and not np.isnan(np.log(self.flags["pp_alpha"](result.x[3]))):
                    use_log_log += np.log(self.flags["pp_alpha"](result.x[3]))

                if not np.isinf(np.log(self.flags["pp_beta"](result.x[4]))) and not np.isnan(np.log(self.flags["pp_beta"](result.x[4]))):
                    use_log_log += np.log(self.flags["pp_beta"](result.x[4]))

                self.results["use_log_lik"] = use_log_log
                self.results["AIC"] = 2 * len(result.x) + 2 * use_log_log
                self.results["BIC"] = 0.5 * len(result.x) * np.log(180) + use_log_log

            else:
                n_unchanged_trials += 1

        self.display_results()


    def display_results(self):
        """Method that displays the results of the fitting procedure"""
        if self.results["n_log_lik"] is np.inf:  # makes sure that the output is valid
            return

        print(f"---- RESULTS----")
        for key, value in self.results.items():
            print(f"{key}:{' ' * (25 - len(key))}{value}")


