from src.hybird_model import Model
from src.loader import Loader

def main():
    # loader = Loader()
    # print(loader.load_precomputed(1, "A"))
    simulate_subject(0)

def simulate_subject(subj_idx : int):
    loader = Loader()

    subject_data = loader.load_subject(subj_idx)

    subject_id = subject_data["userID"]

    precomputed_data = loader.load_precomputed(subj_idx, subject_id)

    print(subject_id)
    print(subject_data)
    print(precomputed_data)
    # fit model


# TODO / Notes
# - change every use of FLAGS to subscript rather than .
# - change every instance of FLAGS to be self.flags



if __name__ == '__main__':
    main()