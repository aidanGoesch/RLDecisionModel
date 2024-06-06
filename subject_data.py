# SubjectData class that stores data associated with each subject

class SubjectData:
    def __init__(self):
        self.trial_rec = None
        self.mem_rec = None

        self.choice_blocks = None
        self.invalid_probe_trials = None    # type = list
        self.mem_probe_trials = None        # type = set

    def is_empty(self):
        return (self.trial_rec is None and self.mem_rec is None and self.choice_blocks is None and
                self.invalid_probe_trials is None and self.mem_probe_trials is None)

    def load(self, file_path=""):   # load data from an input file
        if file_path == "":
            return False
        else:
            return True

