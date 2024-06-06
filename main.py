# import task_vars
from types import SimpleNamespace

TASK_VARS = SimpleNamespace()
TASK_VARS.MODEL_SAMPLER = 0
TASK_VARS.MODEL_TD = 1
TASK_VARS.MODEL_HYBRID = 2
TASK_VARS.DECAYTYPE_COMBINED = 0
TASK_VARS.DECAYTYPE_BYOPTION = 1
TASK_VARS.responseButtons = [1, 2, 3, 4]

TASK_VARS.OPT_DEFAULTS = SimpleNamespace()

# Task structure.
TASK_VARS.OPT_DEFAULTS.num_trials = 360
TASK_VARS.OPT_DEFAULTS.num_bandits = 3
TASK_VARS.OPT_DEFAULTS.num_probeTrials = 60
TASK_VARS.OPT_DEFAULTS.frac_valid_probes = 50/60
TASK_VARS.OPT_DEFAULTS.room_len = 30
TASK_VARS.OPT_DEFAULTS.num_rooms = 6
TASK_VARS.OPT_DEFAULTS.payoff_switch = 10

# Subject characteristics.
TASK_VARS.OPT_DEFAULTS.mem_accuracy = 0.95
TASK_VARS.OPT_DEFAULTS.mem_confident = 0.85
TASK_VARS.OPT_DEFAULTS.subj_alpha = 0.4
TASK_VARS.OPT_DEFAULTS.subj_alpha_mem = 0.2
TASK_VARS.OPT_DEFAULTS.subj_beta = 1
TASK_VARS.OPT_DEFAULTS.subj_ctx_AC = 0.95
TASK_VARS.OPT_DEFAULTS.subj_persev = 0
TASK_VARS.OPT_DEFAULTS.subj_samples = 6

TASK_VARS.OPT_DEFAULTS.accumulate_samples = False

TASK_VARS.OPT_DEFAULTS.mc_samples = 500
TASK_VARS.OPT_DEFAULTS.which_model = TASK_VARS.MODEL_SAMPLER
TASK_VARS.OPT_DEFAULTS.decayType = TASK_VARS.DECAYTYPE_COMBINED

# TASK_VARS["OPT_DEFAULTS"] = {
#     # Task Structure
#     "num_trials": 360,
#     "num_bandits": 3,
#     "num_probe_trials": 60,
#     "frac_valid_probes": 50/60,
#     "room_len": 30,
#     "num_rooms": 6,
#     "payoff_switch": 10,
#
#     # Subject Characteristics
#     "mem_accuracy": 0.95,
#     "mem_confident": 0.85,
#     "subj_alpha": 0.4,                 # learning / decay rate
#     "subj_alpha_mem": 0.2,             # learning / decay rate on reinstatement
#     "subj_beta": 1,
#     "subj_ctx_ac": 0.95,
#     "subj_persev": 0,
#     "subj_samples": 6,
#
#     "accumulate_samples": False,
#
#     "mc_samples": 500,
#     "which_model": TASK_VARS["MODEL_SAMPLER"],
#     "decay_type": TASK_VARS["DECAY_TYPE_COMBINED"]
# }




def main():
    print("Hello World")



# TODO / Notes
# -make a simulation class
# -add invalid probe trials to task_var type = set
# -make sure that indexing is the same
# -generate mem_rec if not simulating
# -find relationship between opts and taskvars.optdefaults



if __name__ == '__main__':
    main()