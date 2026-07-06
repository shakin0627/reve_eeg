"""
adapted from https://github.com/wjq-learning/CBraMod
"""

import argparse
from collections import defaultdict
import os
import pickle

import lmdb
import numpy as np
from einops import rearrange  # noqa
from scipy import signal
import re


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True, help="Root directory")
parser.add_argument("--processed", type=str, required=True, help="Processed data directory")
args = parser.parse_args()

labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])
root_dir = args.root
files = list(os.listdir(root_dir))
files = sorted(files)

def extract_subject_key(fname):
    m = re.match(r"(sub\d+)", fname)
    return m.group(1) if m else fname

subject_to_files = defaultdict(list)
for f in files:
    subject_to_files[extract_subject_key(f)].append(f)

subject_keys = sorted(subject_to_files.keys())

n = len(subject_keys)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train_subjects = subject_keys[:n_train]
val_subjects = subject_keys[n_train:n_train + n_val]
test_subjects = subject_keys[n_train + n_val:]

files_dict = {
    "train": [f for s in train_subjects for f in subject_to_files[s]],
    "val":   [f for s in val_subjects   for f in subject_to_files[s]],
    "test":  [f for s in test_subjects  for f in subject_to_files[s]],
}

# raw_subject_id
raw_subject_id_map = {
    sid: idx
    for idx, sid in enumerate(subject_keys)
}

# domain_id
train_domain_id_map = {fname: k for k, fname in enumerate(files_dict["train"])}

dataset = {
    "train": [],
    "val": [],
    "test": [],
}
by_subject = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
path = args.processed.replace("processed", "processed_cbramod")

os.makedirs(path, exist_ok=True)
db = lmdb.open(path, map_size=6612500172)

for files_key, files_list in files_dict.items():
    for file in files_list:
        subject_key = extract_subject_key(file)
        raw_sid = raw_subject_id_map[subject_key]
        domain_id = train_domain_id_map.get(subject_key, -1)  # train: 0~n_train-1, val/test: -1

        with open(os.path.join(root_dir, file), "rb") as f:
            array = pickle.load(f)
        eeg = signal.resample(array, 6000, axis=2)
        eeg_ = eeg.reshape(28, 32, 30, 200)

        for i, (samples, label) in enumerate(zip(eeg_, labels)):
            for j in range(3):
                sample = samples[:, 10 * j : 10 * (j + 1), :]
                sample_key = f"{file}-{i}-{j}"
                data_dict = {
                    "sample": sample,
                    "label": label,
                    "raw_subject_id": raw_sid,
                    "domain_id": domain_id,
                    "trial_idx": i,
                    "window_idx": j,
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()

                dataset[files_key].append(sample_key)
                by_subject[files_key][raw_sid].append((i, j, sample_key))

by_subject_sorted = {
    split: {
        sid: [k for _, _, k in sorted(entries, key=lambda x: (x[0], x[1]))]
        for sid, entries in subj_dict.items()
    }
    for split, subj_dict in by_subject.items()
}

dataset["train_by_domain"] = by_subject_sorted["train"]
dataset["train_domain_id_map"] = {
    raw_subject_id_map[s]: d for s, d in train_domain_id_map.items()
}
dataset["val_by_subject"] = by_subject_sorted["val"]
dataset["test_by_subject"] = by_subject_sorted["test"]

txn = db.begin(write=True)
txn.put(key="__keys__".encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()