import numpy as np
from types import SimpleNamespace


from src.utils import sign, PARAM_CONSTANTS, FLAGS

NUM_BANDITS = 3
MAX_TRIALS = 180

# Flags for ctx_hybrid = pp_alpha, pp_beta, and pp_betaC
# Params = alpha, beta, beta_c, alpha, beta

class Model:
    def __init__(self, subj_idx: int, input_data: dict, precomputed_data: SimpleNamespace, trial_rec : SimpleNamespace):
        self.input_data = input_data
        self.precomputed = precomputed_data
        self.trial_rec = trial_rec
        self.flags = FLAGS    # load defaults

        self.flags["resetQ"] = False
        self.flags["num_samples"] = 1

        self.flags["choice_rec"] = precomputed_data.choice_rec
        self.flags["combs"] = precomputed_data.combs



    def hybrid_likelihood(self, alpha, beta, beta_c, flags : SimpleNamespace = FLAGS):
        num_samples = flags.num_samples

        # initialize choice trials
        choice_trials = np.array([(x.choice > -1 and x.type == 0) for x in self.trial_rec[:MAX_TRIALS]])   # change this later
        choice_trials = np.argmax(choice_trials, axis=1)

        # Change these later
        alpha_smp = alpha
        beta_smp = beta
        beta_c = beta_c
        alpha_td = alpha
        beta_td = beta

        combs = flags.combs[num_samples]
        choice_rec = flags.choice_rec

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

        n_log_likelihood -= np.log(flags.pp_alpha[alpha_smp])
        n_log_likelihood -= np.log(flags.pp_beta[beta_smp])
        n_log_likelihood -= np.log(flags.pp_beta_C[beta_c])
        n_log_likelihood -= np.log(flags.pp_alpha[alpha_td])
        n_log_likelihood -= np.log(flags.pp_beta[alpha_td])

        return n_log_likelihood


    def fit_model(self, subj_idx):
        print("---- FITTING SUBJECT ----")




