import math
import random
import numpy as np

from types import SimpleNamespace

from subject_data import SubjectData

MODEL_TYPES = ["Sampler", "TD", "Hybrid"]


class Model:
    def __init__(self, sim_type: str, options: SimpleNamespace, task_vars: SimpleNamespace, subject_data: SubjectData = None):
        self.subject_data = subject_data
        assert sim_type in MODEL_TYPES, "ERROR: Invalid Model Type"

        self.sim_type = sim_type
        self.options = None
        self.task_vars = task_vars

        self.parse_args(options, defaults=task_vars["OPT_DEFAULTS"])

    def parse_args(self, opts: dict, defaults: dict):
        """Function that sets default values for options and updates according to options
        passed into the constructor"""
        self.options = defaults
        for key, value in opts.items():
            self.options[key] = value

    def choice_model_context_sample(self, task_vars: dict, verbose: bool):
        """Function that contains the experiment"""
        if self.subject_data is None:
            self.options["simulate_subject"] = True
        else:
            self.options["simulate_subject"] = False


        # COME BACK TO THIS

        self.options["subject_samples"] = math.ceil(self.options["subj_samples"])

        episode_list = [0 for i in range(task_vars["num_trials"] + 1)]

        # create a matrix of dimension num_bandits x num_trials + 1
        bandit_episode_list = [[0 for x in range(self.options["num_bandits"])]
                               for y in range(self.options["num_trials"] + 1)]

        num_bandit_episodes = [0 for i in range(3)]

        trial_idx = 0

        choice_probs = np.array([1 / self.options["num_bandits"] for x in range(self.options["num_trials"])])

        if not self.subject_data.is_empty():
            trial_rec = self.subject_data.trial_rec

            # if mem_rec is None then make it an empty list
            mem_rec = self.subject_data.mem_rec if self.subject_data.mem_rec is not None else []

            task_vars["choice_blocks"] = self.subject_data.choice_blocks
            task_vars["invalid_probe_trials"] = list(sorted(self.subject_data.invalid_probe_trials))
            task_vars["mem_probe_trials"] = (
                self.subject_data.mem_probe_trials.difference(task_vars["invalid_probe_trials"]))
            task_vars["contexts"] = list(range(self.options["num_rooms"]))

        else:
            trial_rec = [0 for i in range(self.options["num_trials"])]
            mem_rec = []

            task_vars["init_payouts"] = [60, 30, 10]
            random.shuffle(task_vars["init_payouts"])
            task_vars["decay_theta"]= task_vars["init_payouts"]
            task_vars["decay_lambda"] = 0.6
            task_vars["drift_sigma"]= 8
            task_vars["drift_noise"]= np.cholesky(np.array([1]) * task_vars["drift_sigma"] ** 2)    # this could be a problem
            task_vars["payoff_bounds"] = [5, 95]
            task_vars["ctx_bump"] = 3

            num_probes = (self.options["payoff_switch"] *
                          self.options["num_rooms"])
            mean_CT = 5
            max_CT = 8
            min_CT = 2

            choice_blocks = -np.ceil(np.log(np.random.rand(num_probes)) / (1 / mean_CT)) + min_CT    # this could also be a problem
            choice_blocks = choice_blocks[choice_blocks > max_CT]

            # trim the generated choice blocks until they sum to the length of the final room
            # and they fit withing (minCT, maxCT)
            # make helper func

            def sign(x : int):
                return math.copysign(1, x) if x != 0 else 0

            while ((sum(choice_blocks) != self.options["num_trials"] / 2)
                    or any(x > max_CT for x in choice_blocks)
                    or any(x < min_CT for x in choice_blocks)):

                i = math.ceil(random.choice(range(num_probes)))  # pick a random block to trim
                choice_blocks[i] = (choice_blocks[i] -
                                    sign(sum(choice_blocks) - self.options["num_trials"] / 2))

                choice_blocks[choice_blocks < min_CT] = min_CT
                choice_blocks[choice_blocks > max_CT] = max_CT

            choice_blocks = [x - 1 for x in choice_blocks]
            task_vars["choice_blocks"] = choice_blocks

            # this could also cause problems
            # Place a memory probe trial at the end of every choice block
            task_vars["mem_probe_trials"] = (np.cumsum([x + 1 for x in task_vars["choice_blocks"]])
                                             + (self.options["room_len"] * self.options["num_rooms"]))

            if verbose:  # debug statement
                print("choicemodel_ctxSample: Generated choice trials lenths, sum:", sum(task_vars["choice_blocks"]),
                      " mean:", np.mean(task_vars["choice_blocks"]))

                print("choice_blocks:", task_vars["choice_blocks"])
                print("mem_probe_trials:", task_vars["mem_probe_trials"])

            trial_nums = set(range(1, self.options["num_trials"] + 1))
            task_vars["choice_trials"] = trial_nums.difference(set(task_vars["mem_probe_trials"]))

            # Shuffle the list of memory probe trials
            random.shuffle(task_vars["mem_probe_trials"])

            # Take the first numInvalidProbes of indexes
            task_vars["num_invalid_probes"] = math.ceil((1 - self.options["frac_valid_probes"])
                                                        * self.options["num_probe_trials"])
            task_vars["invalid_probe_trials"] = (
                list(sorted(task_vars["mem_probe_trials"][:task_vars["num_invalid_probes"]])))

            # come back to this
            task_vars["available_for_mem_probes"] = []
            for i in range(1, self.options["num_rooms"]):
                pass

            # [taskvars.availableForMemProbe(opts.roomLen * bIdx): ((opts.roomLen * bIdx) + opts.payoffSwitch - 1)];
            task_vars["contexts"] = [0 for i in range(self.options["num_trials"])]

            ep = 0    # make sure that it can be seen in the outer scope
            for ci in range(1, self.options["num_rooms"]):
                sp = ((self.options["room_len"] - self.options["payoff_switch"]) +
                        ((ci - 1) * self.options["room_len"]))
                ep = sp + self.options["room_len"]
                task_vars["contexts"][sp + 1 : ep] = ci    # make sure that indexing is the same

            # this shouldn't work - how is it turning into a matrix
            task_vars["contexts"][ep + 1:] = max(task_vars["contexts"]) + 1

            task_vars["payout"] = np.array([np.array([0 for x in range(self.options["num_trials"])])
                                   for y in range(self.options["num_bandits"])])

            task_vars["payout"][:, 0] = task_vars["init_payouts"]

            if len(mem_rec) == 0 and not self.options["simulate_subject"]:    # maybe make a dif container for opts
                pass # generate mem_rec

            for b_idx in range(1, self.options["num_rooms"] + 1):   # iterate through the bandits and do choice trials for each bandit
                if self.options["verbose"]:
                    print(f"choicemodel_ctxSample: Entering room", b_idx)

                for j in range(1, self.options["room_len"] + 1):
                    choice_probs[trial_idx], trial_rec, task_vars = self.do_choice_trial(trial_rec, task_vars)
                    trial_idx += 1


            for cb in range(len(task_vars["choice_blocks"][1])):
                for ct in task_vars["choice_blocks"][cb]:
                    trial_idx += 1
                    choice_probs[trial_idx], trial_rec, task_vars = self.do_choice_trial(trial_rec, task_vars)

                trial_idx += 1
                trial_rec, task_vars = self.do_mem_probe(trial_rec, mem_rec, task_vars)


    def do_choice_trial(self, trial_rec, task_vars: dict) -> tuple[int, int, int]:
        """Function that runs a choice trial"""

        # probability of choosing each bandit on this trial
        cp = [0 for x in range(self.options["num_bandits"])]

        if self.options["simulate_subject"]:   # Simulation
            # Generate payoffs
            if task_vars["trial_idx"]

    def do_mem_probe(self, trial_rec, mem_rec, task_vars: dict) -> tuple[int, int]:
        pass

