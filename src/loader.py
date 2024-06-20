import pathlib as pl
import scipy
import glob


class Loader:
    def __init__(self):
        self.files = None

    def load_subject(self, subj_idx : int):
        """Function that loads transformed subject data"""
        self.files = glob.glob(f'./data/transformed_Data*.mat')

        subject_info = {}
        complete_path = f'{self.files[subj_idx]}'

        subject_id = complete_path.split('_')[-1].split('.')[0]

        subject_mat = scipy.io.loadmat(complete_path, squeeze_me=True, struct_as_record=False)
        # print(subject_mat)
        # print(dir(subject_mat['trialrec'][0]))
        subject_info = {
            'userID': subject_mat['userID'],
            'age': subject_mat['age'],
            'sex': subject_mat['sex'],
            'mst': subject_mat['mst'],
            'ldi': subject_mat['ldi'],
            'pss': subject_mat['pss'],
            'quic': subject_mat['quic'],
            'aprime': subject_mat['aprime'],
            'dprime': subject_mat['dprime'],
            'pseries': subject_mat['pseries'],
            'exp_date': subject_mat['expDate'],
            'trial_rec': [{
                'type': x.type,
                'choice': x.choice,
                'rwdval': x.rwdval,
                'contexts': x.contexts,
                'RT': x.RT,
                'probed_fn': x.probed_fn,
                'probed': x.probed

            } for x in subject_mat['trialrec']]
        }

        return subject_info

    def load_precomputed(self, subj_idx : int, subj_id : str):
        """Function that loads precomputed data"""
        subject_info = {}

        self.files = glob.glob(f'./precomputed/precomputed_sub*.mat')

        complete_path = f'{self.files[subj_idx]}'

        precomputed_mat = scipy.io.loadmat(complete_path, squeeze_me=True, struct_as_record=False)

        subject_info[subj_id] = {
            'combs': precomputed_mat['combs'],
            'choice_rec': precomputed_mat['choicerec'],
        }

        return subject_info


if __name__ == '__main__':
    loader = Loader()
    print(loader.load_subject(0))