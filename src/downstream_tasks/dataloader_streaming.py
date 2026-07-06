import hydra
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
from downstream_tasks.position_utils import load_positions
import pickle
from os.path import join as pjoin

class StreamingEvalDataset(Dataset):
    """
    (subject, trial_idx, window_idx) 
    """
    def __init__(self, path, split, positions=None, electrodes=None, scale_factor=100):
        assert split in ("val", "test"), f"split must be val/test, got {split}"
        self.path = path
        self.scale_factor = scale_factor

        env = lmdb.open(path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=1024)
        with env.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get("__keys__".encode()))
            by_subject = all_keys[f"{split}_by_subject"]  # {raw_sid: [keys], already (trial,window)-sorted}
        env.close()

        self.subject_ids = sorted(by_subject.keys())

        self.flat_keys = []
        self.is_subject_start = []
        self.raw_subject_ids = []
        for sid in self.subject_ids:
            keys = by_subject[sid]
            for pos_in_seq, k in enumerate(keys):
                self.flat_keys.append(k)
                self.is_subject_start.append(pos_in_seq == 0)
                self.raw_subject_ids.append(sid)

        if positions is not None:
            positions = pjoin(path, positions)
        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)
        self.db = None

    def __len__(self):
        return len(self.flat_keys)

    def _init_db(self):
        if self.db is None:
            self.db = lmdb.open(self.path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=1024)

    def __getitem__(self, index):
        self._init_db()
        key = self.flat_keys[index]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))

        return {
            "sample": torch.from_numpy(pair["sample"] / self.scale_factor).float(),
            "label": torch.tensor(pair["label"]).long(),
            "raw_subject_id": torch.tensor(self.raw_subject_ids[index]).long(),
            "is_subject_start": torch.tensor(self.is_subject_start[index]),
            "pos": self.positions,
        }
    
def get_streaming_eval_loader(path, split, positions=None, electrodes=None, scale_factor=100):
    dataset = StreamingEvalDataset(
        path, split, positions=positions, electrodes=electrodes, scale_factor=scale_factor,
    )
    return DataLoader(
        dataset,
        batch_size=1,       
        shuffle=False,      
        num_workers=0,      
        collate_fn=None,    
    )