import math
import random
import numpy as np

from src.utils import sign

MODEL_TYPES = ["Sampler", "TD", "Hybrid"]

TASKVARS = {
    "optdefaults": {
        "numTrials": 360,
        "numBandits": 3,
        "numProbeTrials": 60,
        "fracValidProbes": 50 / 60,
        "roomLen": 30,
        "numRooms": 6,
        "payoffSwitch": 10,
        # Subject characteristics
        "memAccuracy": 0.95,  # dist?
        "memConfident": 0.85,  # dist?
        "subjAlpha": 0.4,  # \alpha = learning rate / decay rate                         # dist? set to fit values?
        "subjAlphaMem": 0.2,  # \alpha_{mem} = learning / decay rate on reinstatements      # dist? set to fit values?
        "subjBeta": 1,  # \beta  = softmax temp                                       # dist? set to fit values?
        "subjCtxAC": 0.95,  # \pi    = context autocorrelation between successive samples # dist? set to fit values?
        "subjPersev": 0,  # Choice stickiness
        "subjSamples": 6,  # # of samples to draw for each choice                        # dist? set to fit values? 0 == use adaptive threshold (XXX unimp)?
        "accumulateSamples": False,
        "mcSamples": 500,
        "whichModel": "MODEL_SAMPLER",  # Placeholder for `taskvars.MODEL_SAMPLER`
        "decayType": "DECAYTYPE_COMBINED",  # Placeholder for `taskvars.DECAYTYPE_COMBINED`
    }
}


# DEPRECATED - NOT IN USE

class Model:
    def __init__(self, options: dict, subject_data: "SubjectData" = None):
        self.subject_data = subject_data

        self.options = None
        self.task_vars = TASKVARS

        self.trial_rec = None

        self.parse_args(options, defaults=TASKVARS["optdefaults"])

    def parse_args(self, opts: dict, defaults: dict):
        """Function that sets default values for options and updates according to options
        passed into the constructor"""
        self.options = defaults
        for key, value in opts.items():
            self.options.__dict__[key] = value

    def choice_model_context_sample(self):
        """Function that contains the experiment"""
        if self.subject_data is None:
            self.options.simulate_subject = True
        else:
            self.options.simulate_subject = False

        if self.options.get("mcSamples") is None:
            if self.options.get("simulateSubj"):
                # Simulating: Just one draw.
                self.options["mcSamples"] = 1
            else:
                # Fitting: Approximate distribution using mcSamples # of samples
                self.options["mcSamples"] = self.task_vars["optdefaults"]["mcSamples"]

        self.options.subject_samples = math.ceil(self.options.subj_samples)

        episode_list = [0 for i in range(self.task_vars.num_trials + 1)]

        # create a matrix of dimension num_bandits x num_trials + 1
        bandit_episode_list = [[0 for x in range(self.options.num_bandits)]
                               for y in range(self.options.num_trials + 1)]

        num_bandit_episodes = [0 for i in range(3)]

        self.task_vars.trial_idx = 0

        choice_probs = np.array([1 / self.options.num_bandits for x in range(self.options.num_trials)])

        if not self.subject_data.is_empty():
            self.trial_rec = self.subject_data.trial_rec

            # if mem_rec is None then make it an empty list
            mem_rec = self.subject_data.mem_rec if self.subject_data.mem_rec is not None else []

            self.task_vars.choice_blocks = self.subject_data.choice_blocks
            self.task_vars.invalid_probe_trials = list(sorted(self.subject_data.invalid_probe_trials))
            self.task_vars.mem_probe_trials = (
                self.subject_data.mem_probe_trials.difference(self.task_vars.invalid_probe_trials))
            self.task_vars.contexts = list(range(self.options.num_rooms))

        else:
            self.trial_rec = [dict() for _ in range(self.options.num_trials)]
            mem_rec = []

            self.task_vars.init_payouts = [60, 30, 10]
            random.shuffle(self.task_vars.init_payouts)
            self.task_vars.decay_theta = self.task_vars.init_payouts
            self.task_vars.decay_lambda = 0.6
            self.task_vars.drift_sigma = 8
            self.task_vars.drift_noise = np.cholesky(np.array([1]) * self.task_vars.drift_sigma ** 2)    # this could be a problem
            self.task_vars.payoff_bounds = [5, 95]
            self.task_vars.ctx_bump = 3

            num_probes = (self.options.payoff_switch *
                          self.options.num_rooms)
            mean_CT = 5
            max_CT = 8
            min_CT = 2

            choice_blocks = -np.ceil(np.log(np.random.rand(num_probes)) / (1 / mean_CT)) + min_CT    # this could also be a problem
            choice_blocks = choice_blocks[choice_blocks > max_CT]

            # trim the generated choice blocks until they sum to the length of the final room
            # and they fit withing (minCT, maxCT)
            # make helper func

            while ((sum(choice_blocks) != self.options.num_trials / 2)
                    or any(x > max_CT for x in choice_blocks)
                    or any(x < min_CT for x in choice_blocks)):

                i = math.ceil(random.choice(range(num_probes)))  # pick a random block to trim
                choice_blocks[i] = (choice_blocks[i] -
                                    sign(sum(choice_blocks) - self.options.num_trials / 2))

                choice_blocks[choice_blocks < min_CT] = min_CT
                choice_blocks[choice_blocks > max_CT] = max_CT

            choice_blocks = [x - 1 for x in choice_blocks]
            self.task_vars.choice_blocks = choice_blocks

            # this could also cause problems
            # Place a memory probe trial at the end of every choice block
            self.task_vars.mem_probe_trials = (np.cumsum([x + 1 for x in self.task_vars.choice_blocks])
                                             + (self.options.room_len * self.options.num_rooms))

            if self.options.verbose:  # debug statement
                print("choicemodel_ctxSample: Generated choice trials lenths, sum:", sum(self.task_vars.choice_blocks),
                      " mean:", np.mean(self.task_vars.choice_blocks))

                print("choice_blocks:", self.task_vars.choice_blocks)
                print("mem_probe_trials:", self.task_vars.mem_probe_trials)

            trial_nums = set(range(1, self.options.num_trials + 1))
            self.task_vars.choice_trials = trial_nums.difference(set(self.task_vars.mem_probe_trials))

            # Shuffle the list of memory probe trials
            random.shuffle(self.task_vars.mem_probe_trials)

            # Take the first numInvalidProbes of indexes
            self.task_vars.num_invalid_probes = math.ceil((1 - self.options.frac_valid_probes)
                                                        * self.options.num_probe_trials)
            self.task_vars.invalid_probe_trials = (
                list(sorted(self.task_vars.mem_probe_trials[:self.task_vars.num_invalid_probes])))

            # come back to this
            self.task_vars.available_for_mem_probes = []
            for i in range(1, self.options.num_rooms):
                pass

            # [taskvars.availableForMemProbe(opts.roomLen * bIdx): ((opts.roomLen * bIdx) + opts.payoffSwitch - 1)];
            self.task_vars.contexts = [0 for i in range(self.options.num_trials)]

            ep = 0    # make sure that it can be seen in the outer scope
            for ci in range(1, self.options.num_rooms):
                sp = ((self.options.room_len - self.options.payoff_switch) +
                        ((ci - 1) * self.options.room_len))
                ep = sp + self.options.room_len
                self.task_vars.contexts[sp + 1 : ep] = ci    # make sure that indexing is the same

            # this shouldn't work - how is it turning into a matrix
            self.task_vars.contexts[ep + 1:] = max(self.task_vars.contexts) + 1

            self.task_vars.payout = np.array([np.array([0 for x in range(self.options.num_trials)])
                                   for y in range(self.options.num_bandits)])

            self.task_vars.payout[:, 0] = self.task_vars.init_payouts

            if len(mem_rec) == 0 and not self.options.simulate_subject:    # maybe make a dif container for opts
                pass # generate mem_rec

            for b_idx in range(1, self.options.num_rooms + 1):   # iterate through the bandits and do choice trials for each bandit
                if self.options.verbose:
                    print(f"choicemodel_ctxSample: Entering room", b_idx)

                for j in range(1, self.options.room_len + 1):
                    choice_probs[self.task_vars.trial_idx] = self.do_choice_trial()
                    self.task_vars.trial_idx += 1

            for cb in range(len(self.task_vars.choice_blocks[1])):
                for ct in self.task_vars.choice_blocks[cb]:
                    self.task_vars.trial_idx += 1
                    choice_probs[self.task_vars.trial_idx] = self.do_choice_trial()

                self.task_vars.trial_idx += 1
                self.do_mem_probe(mem_rec)


    def do_choice_trial(self):
        """Function that runs a choice trial"""

        # probability of choosing each bandit on this trial
        cp = [0 for x in range(self.options.num_bandits)]

        if self.options.simulate_subject:   # Simulation
            # Generate payoffs
            if self.task_vars.trial_idx % self.options.room_len == 0:
                best_opt = self.task_vars.decay_theta.index(max(self.task_vars.decay_theta))    # get the max index of decay theta

                while self.task_vars.decay_theta[best_opt] == max(self.task_vars.decay_theta):   # this is weird
                    random.shuffle(self.task_vars.decay_theta)

            if self.task_vars.trial_idx > 1:
                if self.task_vars.trial_idx % self.options.room_len < self.task_vars.ctx_bump:    # make sure that this is in task_vars
                    decay_lambda_eff = 0.95
                else:
                    decay_lambda_eff = self.task_vars.decay_lambda

                for b in range(self.options.num_bandits):    # iterate through the bandits
                    self.task_vars.payout[b][self.task_vars.trial_idx] = (decay_lambda_eff * self.task_vars.payout[b][self.task_vars.trial_idx - 1] +
                                                                          (1 - decay_lambda_eff) * self.task_vars.decay_theta[b] +
                                                                   np.random.randn() * self.task_vars.drift_noise)

                    if self.task_vars.payout[b][self.task_vars.trial_idx] > self.task_vars.payoff_bounds[1]:
                        self.task_vars.payout[b][self.task_vars.trial_idx] = (self.task_vars.payoff_bounds[1] -
                                                                              (self.task_vars.payout[b][self.task_vars.trial_idx] - self.task_vars.payoff_bounds[1]))

                    if self.task_vars.payout[b][self.task_vars.trial_idx] < self.task_vars.payoff_bounds[0]:
                        self.task_vars.payout[b][self.task_vars.trial_idx] = (self.task_vars.payoff_bounds[0] +
                                                                              (self.task_vars.payoff_bounds[0] - self.task_vars.payout[b][self.task_vars.trial_idx]))

            self.trial_rec[self.task_vars.trial_idx].bandits = self.task_vars.payout[:][self.task_vars.trial_idx]   # change the type of container of trial_rec
            self.trial_rec[self.task_vars.trial_idx].decay_theta = self.task_vars.decay_theta
            self.trial_rec[self.task_vars.trial_idx].contexts = self.task_vars.contexts[self.task_vars.trial_idx]

        # No more simulation

        if self.task_vars.trial_idx == 0:
            chosen_bandit = np.ceil(random.random() * self.options.num_bandits)
            cp = [1 / self.options.num_bandits for x in range(self.options.num_bandits)]

        else:
            for mc_idx in range(self.options.mc_samples):
                if self.options.which_model == "MODEL_SAMPLER":
                    # improvement possible
                    sample_context = -1
                    sample_choice = []    # this may be wrong
                    sample_value = []

                    for sample_idx in range(self.options.subject_samples):
                        got_sample = False
                        while not got_sample:
                            if (sample_context != -1 and random.random() < self.options.subj_ctx_AC and
                                sample_context != max(self.task_vars.contexts) and
                                    sample_context != self.trial_rec[self.task_vars.trial_idx - 1].contexts):

                                if self.options.decay_type == "DECAYTYPE_COMBINED":
                                    # improvement possible
                                    ctx_trials = []
                                    try:
                                        ctx_trials = next((x for x in self.trial_rec if x.contexts == sample_context))  # find the ctx
                                    except StopIteration:
                                        print("This is bad and shouldn't happen")
                                        quit()

                                    sampled_trial = math.ceil(random.random() * len(ctx_trials))

                                    sampled_trial = ctx_trials[sampled_trial]

                                    if len(self.task_vars.episode_list[sampled_trial]) != 0:
                                        got_sample = True
                                        sample_choice[sample_idx] = self.trial_rec[sampled_trial].choice                # i feel like this could cause problems
                                        sample_value[sample_idx] = sign(self.trial_rec[sampled_trial].rwdval) * 2 - 1

                                elif self.options.decay_type == self.task_vars.DECAYTYPE_BYOPTION:
                                    got_sample = True

                                    for bandit_idx in range(self.options.num_bandits):
                                        if not self.task_vars.num_bandit_episodes[bandit_idx]: continue     # what

                                        ctx_trials = next((x for x in self.trial_rec if x.contexts == sample_context and x.choice == bandit_idx))

                                        if len(ctx_trials) == 0:
                                            continue # no samples for this bandit this trial. Skip it

                                        sampled_trial = math.ceil(random.random() * len(ctx_trials))
                                        sampled_trial = ctx_trials[sampled_trial]

                                        sample_choice.append(bandit_idx)
                                        sample_value.append(sign(self.trial_rec[sampled_trial].rwdval) * 2 - 1)
                            else:
                                if self.options.decay_type == self.task_vars.DECAYTYPE_COMBINED:
                                    # Draw one sample
                                    sample_trial_probs = np.array([self.options.subj_alpha ** x for x in range(self.task_vars.trial_idx)])
                                    sample_trial_probs = sample_trial_probs / np.sum(sample_trial_probs)

                                    tmp_logical_arr = random.random() < np.cumsum(sample_trial_probs)
                                    sampled_trial = self.task_vars.trial_idx - (np.argmax(tmp_logical_arr))      # stops after finding the first thing that fits the predicate

                                    if len(sampled_trial) != 0 and len(self.task_vars.episode_list[sampled_trial]) != 0:
                                        got_sample = True
                                        sample_context = self.task_vars.episode_list[sampled_trial].contexts
                                        sample_choice[sample_idx] = self.task_vars.episode_list[sampled_trial].choice
                                        sample_value[sample_idx] = sign(self.task_vars.episode_list[sampled_trial].rwdval) * 2 - 1

                                elif self.options.decay_type == self.task_vars.DECAYTYPE_BYOPTION:
                                    got_sample = True
                                    sample_indices = []
                                    sample_context_candidates = []

                                    for bandit_idx in range(self.options.num_bandits):
                                        if not self.task_vars.num_bandit_episodes[bandit_idx]: continue

                                        sample_trial_probs = np.array([self.options.subj_alpha ** x for x in range(self.task_vars.num_bandit_episodes[bandit_idx] - 1)])
                                        sample_trial_probs = sample_trial_probs / np.sum(sample_trial_probs)

                                        tmp_logical_arr = random.random() < np.cumsum(sample_trial_probs)
                                        sampled_trial = self.task_vars.num_bandit_episodes[bandit_idx] - (np.argmax(tmp_logical_arr))

                                        sample_choice.append(bandit_idx)
                                        sample_value.append(sign(self.task_vars.bandit_episode_list[sampled_trial][sampled_trial].rwdval) * 2 - 1)
                                        sample_indices.append(len(sample_choice))
                                        sample_context_candidates.append(self.task_vars.bandit_episode_list[bandit_idx][sampled_trial].contexts)

                                    diff_ctx = np.where((np.array(sample_context_candidates) != sample_context) & (np.array(sample_context_candidates) != -1))[0]

                                    if len(diff_ctx) != 0:
                                        sample_context_candidates = sample_context_candidates[diff_ctx]
                                        sample_context = sample_context_candidates[math.ceil(random.random()*len(diff_ctx))]

                                        if self.options.veryverbose:
                                            print("choicemodel_ctxSample: Trial", self.task_vars.trial_idx,
                                                 "MCsample", mc_idx, "switch to context:", sample_context)

                                        if self.options.veryverbose:
                                            print("DEBUG")
                        if self.options.veryverbose:
                            print("DEBUG")

                    bandit_value = np.array([0 for x in range(self.options.num_bandits)])

                    for bandit_idx in range(self.options.num_bandits):
                        if self.options.accumulate_samples:
                            bandit_value[bandit_idx] = (sum(np.array(sample_value)[sample_choice == bandit_idx]) -
                                                        sum(0.5 * np.array(sample_value)[sample_choice != bandit_idx]))
                        else:
                            bandit_value[bandit_idx] = np.mean(np.array(sample_value)[sample_choice == bandit_idx])

                    last_choice_trial = self.task_vars.trial_idx
                    if len(self.trial_rec[last_choice_trial].bandits) == 0:
                        last_choice_trial = self.task_vars.trial_idx - 1

                    persev = ([np.array(self.trial_rec[last_choice_trial].choice) == np.array([x for x in range(self.options.num_bandits)])] * 2) - 1
                    persev = self.options.subj_persev * persev

                    if not self.options.simulate_subject:
                        chosen_bandit = self.trial_rec[self.task_vars.trial_idx].choice
                        if chosen_bandit == 0:
                            persev = [0 for x in range(self.options.num_bandits)]

                    # Compute choice probabilities
                    if self.options.accumulate_samples:
                        bandit_probs = bandit_value == max(bandit_value)
                    else:
                        denom = np.exp(persev[0] + self.options.subj_beta * bandit_value[0]) + \
                            np.exp(persev[1] + self.options.subj_beta * bandit_value[1]) + \
                            np.exp(persev[2] + self.options.subj_beta * bandit_value[2])

                        bandit_probs = np.exp(np.array(persev) + self.options.subj_beta * bandit_value)/denom

                    cp = cp + bandit_probs

                    if self.options.veryverbose:
                        print("DEBUG")

        chosen_bandit = -1   # initialize in higher scope to make it accessible
        if not self.options.simulate_subject:
            chosen_bandit = self.trial_rec[self.task_vars.trial_idx].choice

        if not self.options.simulate_subject and chosen_bandit == 0:
            cp = 1 / self.options.num_bandits
        else:
            cp = cp / self.options.mc_samples


        if self.options.simulate_subject:
            chosen_bandit = np.argmax(random.random() < np.cumsum(cp))
            self.trial_rec[self.task_vars.trial_idx].choice = chosen_bandit

            self.trial_rec[self.task_vars.trial_idx].RT = -1       # this is not used rn?

            if (self.task_vars.trial_idx > self.task_vars.ctx_bump and
                self.trial_rec[self.task_vars.trial_idx].contexts != self.trial_rec[self.task_vars.trial_idx - self.task_vars.ctx_bumpp].contexts and
                self.trial_rec[self.task_vars.trial_idx - self.task_vars.ctx_bump].contexts > -1):
                fav_opt = self.task_vars.decay_theta.index(max(self.task_vars.decay_theta))
                self.trial_rec[self.task_vars.trial_idx].bandits[fav_opt] = 100
                self.trial_rec[self.task_vars.trial_idx].decay_theta[fav_opt] = 100
                is_rewarded = chosen_bandit == fav_opt
            else:
                is_rewarded = random.random() < (self.task_vars.payout[chosen_bandit][self.task_vars.trial_idx] / 100)

            self.trial_rec[self.task_vars.trial_idx].rwdval = is_rewarded * 10
            self.trial_rec[self.task_vars.trial_idx].probed = self.task_vars.trial_idx


        if chosen_bandit != 0:
            cp = cp[chosen_bandit]

            self.task_vars.num_bandit_episodes[chosen_bandit] += 1
            self.task_vars.bandit_episode_list[chosen_bandit][self.task_vars.num_bandit_episodes[chosen_bandit]] = self.trial_rec[self.task_vars.trial_idx]
            self.task_vars.episode_list[self.task_vars.trial_idx] = self.trial_rec[self.task_vars.trial_idx]
        else:
            cp = 1 / self.options.num_bandits

        return cp, self.trial_rec, self.task_vars

    def do_mem_probe(self, probe_type, probe_data):
        """
        Executes a memory probe based on the given probe type and data.

        Args:
            probe_type (str): The type of memory probe ('free_recall', 'cued_recall', etc.).
            probe_data (dict): Relevant data for the memory probe, including cues and responses.

        Returns:
            dict: The result of the memory probe, including accuracy, response time, and other metrics.
        """
        # Initialize response metrics
        response = {
            "accuracy": 0.0,
            "response_time": None,
            "errors": [],
        }

        # Validate probe type
        valid_probe_types = {"free_recall", "cued_recall", "recognition"}
        if probe_type not in valid_probe_types:
            response["errors"].append(f"Invalid probe type: {probe_type}")
            return response

        try:
            # Process based on probe type
            if probe_type == "free_recall":
                response["accuracy"] = self._process_free_recall(probe_data)
            elif probe_type == "cued_recall":
                response["accuracy"] = self._process_cued_recall(probe_data)
            elif probe_type == "recognition":
                response["accuracy"], response["response_time"] = self._process_recognition(probe_data)

        except Exception as e:
            response["errors"].append(str(e))

        return response

    def _process_free_recall(self, probe_data):
        """
        Processes a free recall memory probe.

        Args:
            probe_data (dict): Data for the free recall probe.

        Returns:
            float: Accuracy of the free recall.
        """
        # Simulate free recall logic (placeholder)
        recalled_items = probe_data.get("recalled_items", [])
        target_items = probe_data.get("target_items", [])

        if not target_items:
            raise ValueError("Target items missing for free recall.")

        correct_recall = sum(1 for item in recalled_items if item in target_items)
        return correct_recall / len(target_items)

    def _process_cued_recall(self, probe_data):
        """
        Processes a cued recall memory probe.

        Args:
            probe_data (dict): Data for the cued recall probe.

        Returns:
            float: Accuracy of the cued recall.
        """
        # Simulate cued recall logic (placeholder)
        cues = probe_data.get("cues", {})
        responses = probe_data.get("responses", {})

        if not cues or not responses:
            raise ValueError("Cues or responses missing for cued recall.")

        correct = sum(1 for cue, response in responses.items() if cue in cues and cues[cue] == response)
        return correct / len(cues)

    def _process_recognition(self, probe_data):
        """
        Processes a recognition memory probe.

        Args:
            probe_data (dict): Data for the recognition probe.

        Returns:
            tuple: Accuracy and response time of the recognition probe.
        """
        # Simulate recognition logic (placeholder)
        presented_items = probe_data.get("presented_items", [])
        response_items = probe_data.get("response_items", [])

        if not presented_items:
            raise ValueError("Presented items missing for recognition.")

        correct = sum(1 for item in response_items if item in presented_items)
        accuracy = correct / len(presented_items)

        response_time = probe_data.get("response_time", None)  # Placeholder for response time calculation
        return accuracy, response_time


if __name__ == "__main__":
    model = Model()

