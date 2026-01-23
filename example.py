import os
import lmdb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import zipfile
import random
import collections
import copy
import numpy as np

# Import NTA augmentation
from augmentations.nta_augmentation import nucleotide_augment
# Import Residue Masking augmentations
from augmentations.residue_masking import mask_residues, conservative_mask_residues
# Import Spider augmentation
from augmentations.spider_augmentation import spider_augment
# Import RSA (Retrieved Sequence Augmentation)
from augmentations.rsa_augmentation import rsa_augment
# Import PreIS augmentation
from augmentations.preis_augmentation import preis_augment
# Import NanaMigu augmentations
from augmentations.nana_augmentation import nana_augment
from augmentations.migu_augmentation import migu_augment
# Import IMAEN augmentation
from augmentations.imaen import imaen_simple

# Unzip the yeast_ppi.zip file
zip_path = "/content/yeast_ppi.zip"
extract_path = "/content"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation dictionaries and functions
AMINO_ACID_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_LIST_TO_CODON_LIST = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAU', 'AAC'],
    'D': ['GAU', 'GAC'],
    'C': ['UGU', 'UGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],
    'H': ['CAU', 'CAC'],
    'I': ['AUU', 'AUC', 'AUA'],
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],
    'K': ['AAA', 'AAG'],
    'M': ['AUG'],
    'F': ['UUU', 'UUC'],
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],
    'W': ['UGG'],
    'Y': ['UAU', 'UAC'],
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],
    '*': ['UAA', 'UAG', 'UGA'],
}
CODON_LIST_TO_AMINO_ACID = {codon: aa for aa, codons in AMINO_ACID_LIST_TO_CODON_LIST.items() for codon in codons}

# Augmentation functions
def crop_random_segment(sequence, residue_len):
    seq_len = len(sequence)
    if seq_len == 0:
        return sequence
    crop_len = max(1, int(residue_len * seq_len))
    start = random.randint(0, max(0, seq_len - crop_len))
    return sequence[start:start + crop_len]

def delete_random_residues(sequence, residue_len):
    return [res for res in sequence if random.random() > residue_len]

def reverse_sequence(sequence, residue_len=None):
    return list(reversed(sequence))

def shuffle_random_segment(sequence, residue_len):
    seq_len = len(sequence)
    if seq_len < 2:
        return sequence
    seg_len = max(1, int(residue_len * seq_len))
    start = random.randint(0, max(0, seq_len - seg_len))
    segment = sequence[start:start + seg_len]
    random.shuffle(segment)
    sequence = sequence.copy()
    sequence[start:start + seg_len] = segment
    return sequence

def cut_and_shuffle(sequence, residue_len):
    seq_len = len(sequence)
    if seq_len < 2:
        return sequence
    num_cuts = max(1, int(residue_len * 10))
    cut_points = sorted(random.sample(range(1, seq_len), min(num_cuts, seq_len-1))) + [seq_len]
    segments = [sequence[start:end] for start, end in zip([0] + cut_points[:-1], cut_points)]
    random.shuffle(segments)
    return [res for seg in segments for res in seg]

def subsequence_shuffle(sequence, residue_len):
    seq_len = len(sequence)
    if seq_len < 2:
        return sequence
    num_parts = max(1, int(residue_len * 10))
    cut_points = sorted(random.sample(range(1, seq_len), min(num_parts, seq_len-1))) + [seq_len]
    segments = [sequence[start:end] for start, end in zip([0] + cut_points[:-1], cut_points)]
    selected_segments = random.sample(segments, min(len(segments), num_parts))
    return [res for seg in selected_segments for res in seg]

def insert_random_residues(sequence, residue_len):
    sequence = sequence.copy()
    seq_len = len(sequence)
    num_insertions = max(0, int(residue_len * seq_len))
    for _ in range(num_insertions):
        pos = random.randint(0, len(sequence))
        sequence.insert(pos, random.choice(AMINO_ACID_LIST))
    return sequence

def substitute_random_residues(sequence, residue_len):
    sequence = sequence.copy()
    seq_len = len(sequence)
    num_subs = max(0, int(residue_len * seq_len))
    for _ in range(num_subs):
        pos = random.randint(0, seq_len - 1)
        sequence[pos] = random.choice(AMINO_ACID_LIST)
    return sequence

def swap_random_residues(sequence, residue_len):
    sequence = sequence.copy()
    seq_len = len(sequence)
    num_swaps = max(0, int(residue_len * seq_len))
    for _ in range(num_swaps):
        if seq_len < 2:
            break
        i, j = random.sample(range(seq_len), 2)
        sequence[i], sequence[j] = sequence[j], sequence[i]
    return sequence

def back_translation_substitute(seq, residue_len):
    mRNA = []
    for aa in seq:
        if aa in AMINO_ACID_LIST_TO_CODON_LIST:
            mRNA.extend(list(random.choice(AMINO_ACID_LIST_TO_CODON_LIST[aa])))

    if not mRNA:
        return seq

    mRNA_len = len(mRNA)
    num_subs = max(0, int(residue_len * mRNA_len))
    for _ in range(num_subs):
        pos = random.randint(0, mRNA_len - 1)
        mRNA[pos] = random.choice(['A', 'U', 'C', 'G'])

    codons = ["".join(mRNA[i:i+3]) for i in range(0, len(mRNA), 3)]
    aa_seq = []
    for c in codons:
        if len(c) == 3:
            aa = CODON_LIST_TO_AMINO_ACID.get(c, 'X')
            if aa in AMINO_ACID_LIST:
                aa_seq.append(aa)
    return aa_seq

# All augmentation functions in a list
AUGMENTATION_FUNCTIONS = [
    crop_random_segment,
    delete_random_residues,
    reverse_sequence,
    shuffle_random_segment,
    cut_and_shuffle,
    subsequence_shuffle,
    insert_random_residues,
    substitute_random_residues,
    swap_random_residues,
    back_translation_substitute,
    nucleotide_augment,  # NTA: Nucleotide Augmentation (Minot & Reddy 2022)
    mask_residues,  # Residue Masking (MLM-style, inspired by ProtBERT/ESM)
    conservative_mask_residues,  # Conservative Masking (preserves chemical properties)
    spider_augment,  # Spider: Random substitution + insertion (Spider neurotoxic peptide paper)
    rsa_augment,  # RSA: Retrieved Sequence Augmentation (Chang et al. 2023)
    preis_augment,  # PreIS: Supervised Data Augmentation (simplified self-mixing)
    nana_augment,  # NaNa: Novel Augmentation of New Node Attributes (NanaMigu)
    migu_augment,  # MiGu: Molecular Interactions and Geometric Upgrading (NanaMigu)
    imaen_simple,  # IMAEN: Interpretable Molecular Augmentation (property-aware substitution)
]

# Dataset class with augmentation
class LMDBProteinDataset(Dataset):
    def __init__(self, lmdb_path, max_length=512, augment=False,
                 augmentation_intensity=0.1, augmentation_prob=0.5, policy=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.max_length = max_length
        self.augment = augment
        self.augmentation_intensity = augmentation_intensity
        self.augmentation_prob = augmentation_prob
        self.policy = policy
        self.valid_keys = []

        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    data = pickle.loads(value)
                    if isinstance(data, dict) and all(k in data for k in ['primary_1', 'primary_2', 'interaction']):
                        self.valid_keys.append(key)
                except:
                    continue

        print(f"Loaded {len(self.valid_keys)} valid samples from {lmdb_path}")

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        key = self.valid_keys[idx]
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))

        primary_1 = data['primary_1']
        primary_2 = data['primary_2']

        # Apply augmentation to each sequence independently
        if self.augment:
            primary_1 = self._apply_augmentation(primary_1)
            primary_2 = self._apply_augmentation(primary_2)

        seq1 = self.encode_seq(primary_1)
        seq2 = self.encode_seq(primary_2)
        label = torch.tensor(data['interaction'], dtype=torch.float)
        return (seq1, seq2), label

    def _apply_augmentation(self, seq_str):
        """Apply random augmentation to a sequence string"""
        if self.policy:
            # Apply policy-based augmentation
            return self._apply_augmentation_policy(seq_str)
        else:
            # Apply standard augmentation
            return self._apply_standard_augmentation(seq_str)

    def _apply_standard_augmentation(self, seq_str):
        """Apply standard random augmentation"""
        if random.random() > self.augmentation_prob:
            return seq_str

        seq_list = list(seq_str)
        try:
            aug_func = random.choice(AUGMENTATION_FUNCTIONS)
            augmented_list = aug_func(seq_list, self.augmentation_intensity)
            # Ensure we have at least 5 residues after augmentation
            if len(augmented_list) >= 5:
                return ''.join(augmented_list)
            return seq_str
        except Exception as e:
            print(f"Augmentation error: {str(e)}")
            return seq_str

    def _apply_augmentation_policy(self, seq_str):
        """Apply policy-based augmentation"""
        seq_list = list(seq_str)
        if not self.policy:
            return seq_str

        # Randomly select a sub-policy from the policy
        sub_policy = random.choice(self.policy)

        # Apply each operation in the sub-policy sequentially
        for operation in sub_policy:
            aug_func, p, lam = operation
            if random.random() < p:
                try:
                    augmented_list = aug_func(seq_list, lam)
                    # Skip if augmentation makes sequence too short
                    if len(augmented_list) < 5:
                        continue
                    seq_list = augmented_list
                except Exception as e:
                    print(f"Policy augmentation error: {str(e)}")

        return ''.join(seq_list)

    def encode_seq(self, seq):
        vocab = {aa: i+1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        encoded = [vocab.get(aa, 0) for aa in seq[:self.max_length]]
        padded = encoded + [0] * (self.max_length - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

# Collate function
def collate_fn(batch):
    seqs, labels = zip(*batch)
    seq1_batch, seq2_batch = zip(*seqs)
    return torch.stack(seq1_batch), torch.stack(seq2_batch), torch.stack(labels)

# LSTM Model
class ProteinInteractionLSTM(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, vocab_size=21):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # use final hidden state

    def forward(self, seq1, seq2):
        h1 = self.encode(seq1)
        h2 = self.encode(seq2)
        combined = torch.cat([h1, h2], dim=1)
        return self.fc(combined).squeeze(1)

# Evaluation function
def evaluate(model, loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for seq1, seq2, labels in loader:
            seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
            preds = model(seq1, seq2) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
    return correct / total

# Training function for Stage 1
def train_stage1(epochs=5):
    # Training dataset with uniform augmentation
    train_ds = LMDBProteinDataset(
        "/content/yeast_ppi/yeast_ppi_train.lmdb",
        augment=True,
        augmentation_intensity=0.1,
        augmentation_prob=0.7
    )

    # Validation and test without augmentation
    val_ds = LMDBProteinDataset("/content/yeast_ppi/yeast_ppi_valid.lmdb")
    test_ds = LMDBProteinDataset("/content/yeast_ppi/yeast_ppi_test.lmdb")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn)

    model = ProteinInteractionLSTM().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq1, seq2, labels in tqdm(train_loader, desc=f"Stage1 Epoch {epoch + 1}"):
            seq1, seq2, labels = seq1.to(DEVICE), seq2.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seq1, seq2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        print(f"Stage1 Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

    # Test best model
    test_acc = evaluate(best_model, test_loader)
    print(f"Stage1 Test Accuracy: {test_acc:.4f}")

    return best_model

# Training function for Stage 2 (Policy Search)
def train_stage2(shared_model, num_policies=10, sub_policies_per_policy=5, finetune_epochs=1):
    # Create validation dataset (no augmentation)
    val_ds = LMDBProteinDataset("/content/yeast_ppi/yeast_ppi_valid.lmdb")
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn)

    # Define candidate parameters
    p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    lambda_values = [0.05, 0.1, 0.2, 0.3, 0.4]

    # Generate random policies
    candidate_policies = []
    for _ in range(num_policies):
        policy = []
        for _ in range(sub_policies_per_policy):
            sub_policy = []
            # Each sub-policy has 2 operations
            for _ in range(2):
                aug_func = random.choice(AUGMENTATION_FUNCTIONS)
                p = random.choice(p_values)
                lam = random.choice(lambda_values)
                sub_policy.append((aug_func, p, lam))
            policy.append(sub_policy)
        candidate_policies.append(policy)

    best_policy = None
    best_val_acc = 0.0
    best_model = None

    print(f"Starting Stage 2: Testing {num_policies} policies with {finetune_epochs} finetune epochs each")

    for i, policy in enumerate(candidate_policies):
        print(f"\nEvaluating policy {i+1}/{num_policies}")

        # Create training dataset with current policy
        train_ds = LMDBProteinDataset(
            "/content/yeast_ppi/yeast_ppi_train.lmdb",
            augment=True,
            policy=policy
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

        # Clone the shared model
        model = copy.deepcopy(shared_model)
        model.to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Smaller LR for fine-tuning

        # Fine-tune for a few epochs
        for epoch in range(finetune_epochs):
            model.train()
            total_loss = 0
            for seq1, seq2, labels in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}"):
                seq1, seq2, labels = seq1.to(DEVICE), seq2.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(seq1, seq2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate on validation set
        val_acc = evaluate(model, val_loader)
        print(f"Policy {i+1} Val Acc: {val_acc:.4f}")

        # Update best policy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_policy = policy
            best_model = copy.deepcopy(model)

    return best_model, best_policy, best_val_acc

# Full APA training pipeline
def train_apa():
    # Stage 1: Train weight-shared model with uniform augmentation
    print("=" * 50)
    print("Starting Stage 1: Training weight-shared model")
    print("=" * 50)
    shared_model = train_stage1(epochs=5)

    # Stage 2: Policy search
    print("\n" + "=" * 50)
    print("Starting Stage 2: Policy search")
    print("=" * 50)
    best_model, best_policy, best_val_acc = train_stage2(
        shared_model,
        num_policies=10,
        sub_policies_per_policy=5,
        finetune_epochs=1
    )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Evaluate on test set
    test_ds = LMDBProteinDataset("/content/yeast_ppi/yeast_ppi_test.lmdb")
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=collate_fn)
    test_acc = evaluate(best_model, test_loader)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    return best_model, best_policy

# Run APA training
best_model, best_policy = train_apa()