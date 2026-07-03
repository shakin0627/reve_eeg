"""
Common dataloaders for different tasks.
"""

from collections import defaultdict
import os
import pickle
from os.path import join as pjoin
from random import random

import hydra
import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
from downstream_tasks.position_utils import load_positions

################## Sampler ###################
class PKBatchSampler(Sampler):
    def __init__(self, domain_ids, P=16, K=4, seed=0):
        self.P = P
        self.K = K
        self.epoch = 0
        self.base_seed = seed

        self.domain_to_indices = defaultdict(list)
        for idx, d in enumerate(domain_ids):
            self.domain_to_indices[d].append(idx)

        self.domains = list(self.domain_to_indices.keys())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.base_seed + self.epoch)

        domain_pools = {
            d: self._shuffled(idxs, rng) for d, idxs in self.domain_to_indices.items()
        }
        domain_cursor = {d: 0 for d in self.domains}

        def remaining_batches(d):
            return (len(domain_pools[d]) - domain_cursor[d]) // self.K

        active_domains = [d for d in self.domains if remaining_batches(d) > 0]

        while len(active_domains) >= self.P:
            chosen = rng.sample(active_domains, self.P)
            batch = []
            for d in chosen:
                start = domain_cursor[d]
                batch.extend(domain_pools[d][start:start + self.K])
                domain_cursor[d] += self.K
            rng.shuffle(batch)
            yield batch

            active_domains = [d for d in self.domains if remaining_batches(d) > 0]

    def _shuffled(self, idxs, rng):
        idxs = idxs.copy()
        rng.shuffle(idxs)
        return idxs

    def __len__(self):
        total = sum(len(v) // self.K for v in self.domain_to_indices.values())
        return total // self.P
    
################## Datasets ##################


class LMDBDataset(Dataset):
    def __init__(self, path, positions=None, electrodes=None, mode="train", scale_factor=100):
        super(LMDBDataset, self).__init__()
        self.path = path
        self.scale_factor = scale_factor
        self.mode = mode

        env = lmdb.open(path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=1024)
        with env.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get("__keys__".encode()))
            self.keys = all_keys[mode]

            if mode == "train":
                by_domain = all_keys["train_by_domain"]         # {raw_sid: [keys...]}
                domain_id_map = all_keys["train_domain_id_map"]  # {raw_sid: domain_id}

                key_to_domain = {
                    k: domain_id_map[raw_sid]
                    for raw_sid, keys_ in by_domain.items()
                    for k in keys_
                }
                self.domain_ids = [key_to_domain[k] for k in self.keys]
            else:
                self.domain_ids = None
        env.close()

        if positions is not None:
            positions = pjoin(path, positions)

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)
        self.db = None

    def __len__(self):
        return len(self.keys)

    def _init_db(self):
        if self.db is None:
            self.db = lmdb.open(self.path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=1024)

    def __getitem__(self, index):
        self._init_db()
        assert self.db is not None, "LMDB environment not initialized"
        key = self.keys[index]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair["sample"]
        label = pair["label"]

        ret = {
            "sample": data / self.scale_factor,
            "label": label,
            "domain_id": pair["domain_id"],
        }
        return ret

    def _to_tensor(self, data):
        return torch.from_numpy(data).float()

    def collate(self, batch):
        x_data = np.array([x["sample"] for x in batch])
        y_label = np.array([x["label"] for x in batch])
        domain_id = np.array([x["domain_id"] for x in batch])
        N = len(batch)
        positions = self.positions.repeat(N, 1, 1)
        return {
            "sample": self._to_tensor(x_data),
            "label": self._to_tensor(y_label).long(),
            "domain_id": torch.from_numpy(domain_id).long(),
            "pos": positions,
        }


class NeuroLMDataset(Dataset):
    def __init__(self, path, mode, positions=None, electrodes=None, scale_factor=100):
        super(NeuroLMDataset, self).__init__()

        self.scale_factor = scale_factor
        self.path = pjoin(path, mode)
        ls = [f for f in os.listdir(self.path) if f.endswith(".pkl")]
        ls = [pjoin(self.path, f) for f in ls]
        self.files = sorted(ls)

        print(f"Found {len(self.files)} files in {self.path}")

        self.positions = load_positions(positions_path=positions, electrode_names=electrodes)

    def __len__(self):
        return len(self.files)

    def _to_tensor(self, data):
        return torch.from_numpy(data).float()

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            sample = pickle.load(f)
        X = sample["X"]
        Y = int(sample["y"])

        return {
            "sample": self._to_tensor(X / self.scale_factor),
            "label": torch.tensor(Y).long().unsqueeze(0),
            "pos": self.positions,
        }

    def collate(self, batch):
        return {
            "sample": torch.stack([x["sample"] for x in batch]),
            "label": torch.tensor([x["label"] for x in batch]),
            "pos": self.positions.repeat(len(batch), 1, 1),
        }


###############################################################################################


def get_data_loaders(config, loader_config, rank=None) -> dict[str, DataLoader]:
    """
    Get data loaders for training, validation, and testing.
    Args:
        config: Configuration object containing dataset and batch size.
    Returns:
        dict: Dictionary containing data loaders for train, val, and test.
    """

    splits = config.get("splits", ["train", "val", "test"])

    train_dataset = hydra.utils.instantiate(config.dataset, mode=splits[0])
    val_dataset = hydra.utils.instantiate(config.dataset, mode=splits[1])
    test_dataset = hydra.utils.instantiate(config.dataset, mode=splits[2])

    if rank is None or rank == 0:
        print(f"Train: {len(train_dataset):,} | Valid: {len(val_dataset):,} | Test: {len(test_dataset):,}")
        print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")

    train_sampler = None
    if rank is not None:
        import idr_torch  # noqa: PLC0415
        raise NotImplementedError
        print("Using distributed sampler", rank, idr_torch.size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
            rank=rank,
            num_replicas=idr_torch.size,
        )
    P = getattr(config, "pk_p", 16)
    K = getattr(config, "pk_k", 4)
    train_batch_sampler = PKBatchSampler(
        domain_ids=train_dataset.domain_ids,
        P=P,
        K=K,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,   
        collate_fn=train_dataset.collate,
        **loader_config,
    )

    return {
        "train": train_loader,
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            collate_fn=val_dataset.collate,
            shuffle=False,
            **loader_config,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            collate_fn=test_dataset.collate,
            shuffle=False,
            **loader_config,
        ),
    }
