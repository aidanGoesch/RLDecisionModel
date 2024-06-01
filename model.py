from subject_data import SubjectData

MODEL_TYPES = ["Sampler", "TD", "Hybrid"]


class Model:
    def __init__(self, subject_data: SubjectData, sim_type: str):
        self.subject_data = subject_data
        assert sim_type in MODEL_TYPES, "ERROR: Invalid Model Type"
        self.sim_type = sim_type


    def parse_args(self, task_vars: dict):
        pass

    def do_choice_trial(self, trial_rec, task_vars: dict):
        pass

    def do_mem_probe(self, trial_rec, mem_rec, task_vars: dict):
        pass

