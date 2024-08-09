from src.hybird_model import HybridModel
from src.sampling_model import SamplingModel
from src.loader import Loader

def main():
    # put in for loop to simulate every subject - too slow otherwise
    simulate_subject(1)

def simulate_subject(subj_idx : int, verbose = False):
    """Function that simulates a specific subject that is determined by the
    subject index passed as a parameter."""
    loader = Loader()

    # load in transformed participant specific data
    subject_data = loader.load_subject(subj_idx)

    subject_id = subject_data["userID"]

    # load in precomputed participant data
    precomputed_data = loader.load_precomputed(subj_idx, subject_id)

    print(f"Fitting subject {subject_id}")
    if verbose:
        print(f"Subject Data: {subject_data}")
        print(f"Precomputed Data: {precomputed_data}")

    # Create and fit model
    trial_rec = subject_data["trial_rec"]
    model = SamplingModel(subj_idx, precomputed_data, trial_rec, True, False)
    model.fit()


# TODO / Next Steps
# - Make a writer - class that can write the data to output files
# - add other models


if __name__ == '__main__':
    main()