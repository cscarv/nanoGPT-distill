"""
Jointly prepare the Shakespeare dataset and TheatreClassique dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = "/nobackup/users/scarv/multi-teacher-distillation/data/english/shakespeare/input.txt"
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data_shakespeare = f.read()
print(f"length of Shakespeare dataset in characters: {len(data_shakespeare):,}")

# load the TheatreClassique dataset
theatre_classique_file_path = "/nobackup/users/scarv/multi-teacher-distillation/data/french/TheatreClassique/train_reformatted.txt"
with open(theatre_classique_file_path, 'r', encoding='utf-8') as f:
    data_theatre_classique = f.read()
print(f"length of TheatreClassique dataset in characters: {len(data_theatre_classique):,}")

# truncate TheatreClassique to same length as Shakespeare dataset
if len(data_theatre_classique) > len(data_shakespeare):
    data_theatre_classique = data_theatre_classique[:len(data_shakespeare)]
    print(f"Truncated TheatreClassique dataset to {len(data_theatre_classique):,} characters to match Shakespeare dataset length.")

# combine both datasets
data = data_shakespeare + "\n" + data_theatre_classique
print(f"length of combined dataset in characters: {len(data):,}")

# get all the unique characters that occur in both datasets
chars_joint = sorted(list(set(data)))
vocab_size = len(chars_joint)
print("all the unique characters in both datasets:", ''.join(chars_joint))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars_joint) }
itos = { i:ch for i,ch in enumerate(chars_joint) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n_eng = len(data_shakespeare)
n_fr = len(data_theatre_classique)
n = n_eng + n_fr
print(f"length of English dataset: {n_eng:,}")
print(f"length of French dataset: {n_fr:,}")
print(f"total length of combined dataset: {n:,}")
train_data_eng = data_shakespeare[:int(n_eng*0.9)]
val_data_eng = data_shakespeare[int(n_eng*0.9):]
train_data_fr = data_theatre_classique[:int(n_fr*0.9)]
val_data_fr = data_theatre_classique[int(n_fr*0.9):]

# encode both to integers
train_ids_eng = encode(train_data_eng)
val_ids_eng = encode(val_data_eng)
print(f"English train has {len(train_ids_eng):,} tokens")
print(f"English val has {len(val_ids_eng):,} tokens")
train_ids_fr = encode(train_data_fr)
val_ids_fr = encode(val_data_fr)
print(f"French train has {len(train_ids_fr):,} tokens")
print(f"French val has {len(val_ids_fr):,} tokens")

# export to bin files
train_ids_eng = np.array(train_ids_eng, dtype=np.uint16)
val_ids_eng = np.array(val_ids_eng, dtype=np.uint16)
train_ids_eng.tofile(os.path.join("/nobackup/users/scarv/multi-teacher-distillation/data/english/shakespeare", 'train.bin'))
val_ids_eng.tofile(os.path.join("/nobackup/users/scarv/multi-teacher-distillation/data/english/shakespeare", 'val.bin'))
train_ids_fr = np.array(train_ids_fr, dtype=np.uint16)
val_ids_fr = np.array(val_ids_fr, dtype=np.uint16)
train_ids_fr.tofile(os.path.join("/nobackup/users/scarv/multi-teacher-distillation/data/french/TheatreClassique", 'train.bin'))
val_ids_fr.tofile(os.path.join("/nobackup/users/scarv/multi-teacher-distillation/data/french/TheatreClassique", 'val.bin'))

# union of train and val sets
train_ids_joint = np.concatenate((train_ids_eng, train_ids_fr))
val_ids_joint = np.concatenate((val_ids_eng, val_ids_fr))
# save the joint train and val sets
train_ids_joint.tofile(os.path.join(os.path.dirname(__file__), 'joint_data', 'train.bin'))
val_ids_joint.tofile(os.path.join(os.path.dirname(__file__), 'joint_data', 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta_joint.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
