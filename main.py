# import task_vars

TASK_VARS = {
    "MODEL_SAMPLER": 0,
    "MODEL_TD": 1,
    "MODEL_HYBRID": 2,
    "DECAY_TYPE_COMBINED": 0,
    "DECAY_TYPE_BY_OPTION": 1,
    "RESPONSE_BUTTONS": [1, 2, 3, 4],
}

TASK_VARS["OPT_DEFAULTS"] = {
    # Task Structure
    "num_trials": 360,
    "num_bandits": 3,
    "num_probe_trials": 60,
    "frac_valid_probes": 50/60,
    "room_len": 30,
    "num_rooms": 6,
    "payoff_switch": 10,

    # Subject Characteristics
    "mem_accuracy": 0.95,
    "mem_confident": 0.85,
    "subj_alpha": 0.4,                 # learning / decay rate
    "subj_alpha_mem": 0.2,             # learning / decay rate on reinstatement
    "subj_beta": 1,
    "subj_ctx_ac": 0.95,
    "subj_persev": 0,
    "subj_samples": 6,

    "accumulate_samples": False,

    "mc_samples": 500,
    "which_model": TASK_VARS["MODEL_SAMPLER"],
    "decay_type": TASK_VARS["DECAY_TYPE_COMBINED"]
}




def main():
    print("Hello World")



# TODO / Notes
# -make taskvars a dict with all of the required fields
# -make a simulation class



if __name__ == '__main__':
    main()