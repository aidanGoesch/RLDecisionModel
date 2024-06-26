from src.hybird_model import Model
from src.loader import Loader

def main():
    # loader = Loader()
    # print(loader.load_precomputed(1, "A"))
    simulate_subject(1)

def simulate_subject(subj_idx : int, verbose = False):
    loader = Loader()

    # load in transformed participant specific data
    subject_data = loader.load_subject(subj_idx)

    subject_id = subject_data["userID"]

    # load in precomputed participant data
    precomputed_data = loader.load_precomputed(subj_idx, subject_id)

    print(f"Fitting subject {subject_id}")
    if verbose:
        print(subject_data)
        print(precomputed_data)

    # Create and fit model
    trial_rec = subject_data["trial_rec"]
    model = Model(subj_idx, precomputed_data, trial_rec, True, False)
    model.fit()


# TODO / Notes
# - make better debug statements
# - add comments explaining code


if __name__ == '__main__':
    main()