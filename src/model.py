import numpy as np
from scipy.optimize import minimize

from src.utils import sign, PARAM_CONSTANTS, FLAGS, ITERATIONS, transform_params

NUM_BANDITS = 3
MAX_TRIALS = 180

class Model:
    def __init__(self):
        # stops unnecessary error warnings during running
        np.seterr(divide='ignore')

        self.params = []
        self.num_params = 0

        self.verbose = False
        self.very_verbose = False

        self.results = None
        self.flags = None


    def likelihood(self, params):
        pass

    def fit(self):    # TODO: make this inheritable
        """Fit the model to the data passed into the constructor"""
        start, n_unchanged_trials = 0, 0

        def f(x):
            """Function to be minimized by scipy.minimize"""
            tmp = (transform_params(x, self.params, minimizing=True))

            if self.very_verbose:
                print("x:", x)
                print("t:", tmp)

            return self.likelihood(params=tmp)[0]

        while n_unchanged_trials < ITERATIONS:
            print(f"Trial {n_unchanged_trials + 1} started")
            start += 1

            # pick random starting values for the params
            x_0 = [np.random.uniform(*PARAM_CONSTANTS[self.params[x]][0]) for x in range(self.num_params)]

            # initial parameters
            transformed_x_0 = transform_params(x_0, self.params)

            # options for minimization function - gtol determines how precise the minimization is
            options = {"disp": False, 'gtol': 0.5, 'maxiter': 1000}

            result = minimize(fun=f, x0=transformed_x_0, options=options)

            transformed_xf = transform_params(result.x, self.params, minimizing=True)

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

                n_unchanged_trials = 0  # reset to zero if nLogLik decreases

                # save results into attribute
                self.results["n_log_lik"] = result.fun
                self.results["params"] = result.x
                self.results["transformed_params"] = transformed_xf
                self.results["model"] = "Hybrid"
                self.results["exit flag"] = result.status
                self.results["output"] = result.message

                _, Q, rpe, pc = self.likelihood(params=result.x)

                self.results["run_Q"] = Q
                self.results["pc"] = pc
                self.results["rpe"] = rpe

                use_log_log = result.fun

                if not np.isinf(np.log(self.flags["pp_alpha"](result.x[0]))) and not np.isnan(
                        np.log(self.flags["pp_alpha"](result.x[0]))):
                    use_log_log += np.log(self.flags["pp_alpha"](result.x[0]))

                if not np.isinf(np.log(self.flags["pp_beta"](result.x[1]))) and not np.isnan(
                        np.log(self.flags["pp_beta"](result.x[1]))):
                    use_log_log += np.log(self.flags["pp_beta"](result.x[1]))

                if not np.isinf(np.log(self.flags["pp_beta_c"](result.x[2]))) and not np.isnan(
                        np.log(self.flags["pp_beta_c"](result.x[2]))):
                    use_log_log += np.log(self.flags["pp_beta_c"](result.x[2]))

                if not np.isinf(np.log(self.flags["pp_alpha"](result.x[3]))) and not np.isnan(
                        np.log(self.flags["pp_alpha"](result.x[3]))):
                    use_log_log += np.log(self.flags["pp_alpha"](result.x[3]))

                if not np.isinf(np.log(self.flags["pp_beta"](result.x[4]))) and not np.isnan(
                        np.log(self.flags["pp_beta"](result.x[4]))):
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

