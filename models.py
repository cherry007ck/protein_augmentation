"""
Neural Network Models for Protein Tasks
Supports PPI and Classification tasks with improved architectures
"""

import torch
import torch.nn as nn
import lmdb
import pickle
import random
from torch.utils.data import Dataset
from augmentations.nta_augmentation import nucleotide_augment
from augmentations.residue_masking import simple_mask_residues


# ============================================================================
# Simple Augmentation Functions
# ============================================================================

def crop_sequence(seq: str, crop_size: float = 0.15) -> str:
    """Randomly crop sequence from either end"""
    if len(seq) < 10:
        return seq
    crop_len = max(1, int(len(seq) * crop_size))
    if random.random() < 0.5:
        return seq[crop_len:]  # Crop from start
    else:
        return seq[:-crop_len]  # Crop from end


def substitute_residues(seq: str, sub_rate: float = 0.15) -> str:
    """Randomly substitute amino acids"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq_list = list(seq)
    num_subs = max(1, int(len(seq) * sub_rate))
    positions = random.sample(range(len(seq)), min(num_subs, len(seq)))
    
    for pos in positions:
        current_aa = seq_list[pos]
        # Choose different amino acid
        new_aa = random.choice([aa for aa in amino_acids if aa != current_aa])
        seq_list[pos] = new_aa
    
    return ''.join(seq_list)


def nta_augment(seq: str, intensity: float = 0.15) -> str:
    """Nucleotide augmentation wrapper"""
    seq_list = list(seq)
    try:
        augmented = nucleotide_augment(seq_list, substitution_rate=intensity)
        return ''.join(augmented)
    except:
        return seq


def mask_sequence(seq: str, mask_rate: float = 0.15) -> str:
    """Mask random residues"""
    try:
        masked = simple_mask_residues(seq, mask_rate=mask_rate)
        return masked
    except:
        return seq


# ============================================================================
# Datasets
# ============================================================================

class PPIDataset(Dataset):
    """Protein-Protein Interaction Dataset (Binary Classification)"""
    
    def __init__(self, lmdb_path, max_length=1024, augment=False, 
                 augmentation_intensity=0.15, augmentation_prob=0.7):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            # Try to get num_examples first
            num_examples_bytes = txn.get(b'num_examples')
            if num_examples_bytes:
                try:
                    self.length = pickle.loads(num_examples_bytes)
                except:
                    try:
                        self.length = int(num_examples_bytes.decode())
                    except:
                        self.length = txn.stat()['entries'] - 1  # Subtract metadata key
            else:
                self.length = txn.stat()['entries']
        self.max_length = max_length
        self.augment = augment
        self.augmentation_intensity = augmentation_intensity
        self.augmentation_prob = augmentation_prob
        
        # Augmentation functions
        self.augmentations = [
            lambda seq: crop_sequence(seq, crop_size=self.augmentation_intensity),
            lambda seq: substitute_residues(seq, sub_rate=self.augmentation_intensity),
            lambda seq: nta_augment(seq, intensity=self.augmentation_intensity),
            lambda seq: mask_sequence(seq, mask_rate=self.augmentation_intensity)
        ]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        
        # Handle different key formats
        seq1 = data.get('seq1', data.get('primary_1', data.get('seq', '')))
        seq2 = data.get('seq2', data.get('primary_2', ''))
        label = float(data.get('label', data.get('interaction', 0)))
        
        # Apply augmentation
        if self.augment and torch.rand(1).item() < self.augmentation_prob:
            aug_fn = torch.randint(0, len(self.augmentations), (1,)).item()
            seq1 = self.augmentations[aug_fn](seq1)
            seq2 = self.augmentations[aug_fn](seq2)
        
        return seq1[:self.max_length], seq2[:self.max_length], label


class LocalizationDataset(Dataset):
    """Subcellular Localization Dataset (Multi-class Classification)"""
    
    def __init__(self, lmdb_path, max_length=1024, augment=False,
                 augmentation_intensity=0.15, augmentation_prob=0.7):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            # Try to get num_examples first
            num_examples_bytes = txn.get(b'num_examples')
            if num_examples_bytes:
                try:
                    self.length = pickle.loads(num_examples_bytes)
                except:
                    try:
                        self.length = int(num_examples_bytes.decode())
                    except:
                        self.length = txn.stat()['entries'] - 1
            else:
                self.length = txn.stat()['entries']
        self.max_length = max_length
        self.augment = augment
        self.augmentation_intensity = augmentation_intensity
        self.augmentation_prob = augmentation_prob
        
        # Augmentation functions
        self.augmentations = [
            lambda seq: crop_sequence(seq, crop_size=self.augmentation_intensity),
            lambda seq: substitute_residues(seq, sub_rate=self.augmentation_intensity),
            lambda seq: nta_augment(seq, intensity=self.augmentation_intensity),
            lambda seq: mask_sequence(seq, mask_rate=self.augmentation_intensity)
        ]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        
        # Handle different key formats
        seq = data.get('seq', data.get('primary', data.get('sequence', '')))
        label = data.get('label', data.get('class', data.get('target', 0)))
        
        # Apply augmentation
        if self.augment and torch.rand(1).item() < self.augmentation_prob:
            aug_fn = torch.randint(0, len(self.augmentations), (1,)).item()
            seq = self.augmentations[aug_fn](seq)
        
        return seq[:self.max_length], label


class HomologyDataset(Dataset):
    """Remote Homology Dataset (Multi-class Classification)"""
    
    def __init__(self, lmdb_path, max_length=1024, augment=False,
                 augmentation_intensity=0.15, augmentation_prob=0.7):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            # Try to get num_examples first
            num_examples_bytes = txn.get(b'num_examples')
            if num_examples_bytes:
                try:
                    self.length = pickle.loads(num_examples_bytes)
                except:
                    try:
                        self.length = int(num_examples_bytes.decode())
                    except:
                        self.length = txn.stat()['entries'] - 1
            else:
                self.length = txn.stat()['entries']
        self.max_length = max_length
        self.augment = augment
        self.augmentation_intensity = augmentation_intensity
        self.augmentation_prob = augmentation_prob
        
        # Augmentation functions
        self.augmentations = [
            lambda seq: crop_sequence(seq, crop_size=self.augmentation_intensity),
            lambda seq: substitute_residues(seq, sub_rate=self.augmentation_intensity),
            lambda seq: nta_augment(seq, intensity=self.augmentation_intensity),
            lambda seq: mask_sequence(seq, mask_rate=self.augmentation_intensity)
        ]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        
        # Handle different key formats
        seq = data.get('seq', data.get('primary', data.get('sequence', '')))
        label = data.get('label', data.get('class', data.get('fold', data.get('target', 0))))
        
        # Apply augmentation
        if self.augment and torch.rand(1).item() < self.augmentation_prob:
            aug_fn = torch.randint(0, len(self.augmentations), (1,)).item()
            seq = self.augmentations[aug_fn](seq)
        
        return seq[:self.max_length], label


# ============================================================================
# Collate Functions
# ============================================================================

def collate_fn_ppi(batch):
    """Collate function for PPI dataset"""
    seqs1, seqs2, labels = zip(*batch)
    
    # Convert sequences to indices
    vocab = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    
    def seq_to_indices(seq):
        return [vocab.get(aa, 0) for aa in seq]
    
    # Pad sequences
    max_len = max(max(len(s1) for s1 in seqs1), max(len(s2) for s2 in seqs2))
    
    seq1_padded = []
    seq2_padded = []
    
    for s1, s2 in zip(seqs1, seqs2):
        s1_idx = seq_to_indices(s1)
        s2_idx = seq_to_indices(s2)
        
        # Pad
        s1_idx += [0] * (max_len - len(s1_idx))
        s2_idx += [0] * (max_len - len(s2_idx))
        
        seq1_padded.append(s1_idx)
        seq2_padded.append(s2_idx)
    
    return (
        torch.LongTensor(seq1_padded),
        torch.LongTensor(seq2_padded),
        torch.FloatTensor(labels)
    )


def collate_fn_classification(batch):
    """Collate function for classification datasets"""
    seqs, labels = zip(*batch)
    
    # Convert sequences to indices
    vocab = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    
    def seq_to_indices(seq):
        return [vocab.get(aa, 0) for aa in seq]
    
    # Pad sequences
    max_len = max(len(s) for s in seqs)
    
    seqs_padded = []
    for s in seqs:
        s_idx = seq_to_indices(s)
        s_idx += [0] * (max_len - len(s_idx))
        seqs_padded.append(s_idx)
    
    return (
        torch.LongTensor(seqs_padded),
        torch.LongTensor(labels)
    )


# ============================================================================
# Models
# ============================================================================

class ProteinInteractionLSTM(nn.Module):
    """
    Improved Bidirectional LSTM for Protein-Protein Interaction Prediction
    Architecture: Embedding -> Bi-LSTM -> FC layers -> Sigmoid
    """
    
    def __init__(self, vocab_size=21, embed_dim=64, hidden_dim=128, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Adjust FC input size for bidirectional LSTM
        lstm_output_dim = hidden_dim * 2  # *2 for bidirectional
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, hidden_dim * 4),  # Concatenate both sequences
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def encode_sequence(self, x):
        """Encode a single sequence"""
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        return torch.cat([h_n[-2], h_n[-1]], dim=1)
    
    def forward(self, seq1, seq2):
        """Forward pass with two sequences"""
        enc1 = self.encode_sequence(seq1)
        enc2 = self.encode_sequence(seq2)
        
        # Concatenate encodings
        combined = torch.cat([enc1, enc2], dim=1)
        
        return self.fc(combined).squeeze(-1)


class ProteinClassificationLSTM(nn.Module):
    """
    Improved Bidirectional LSTM for Protein Classification
    Architecture: Embedding -> Bi-LSTM -> FC layers -> Softmax
    """
    
    def __init__(self, vocab_size=21, embed_dim=128, hidden_dim=256, 
                 num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        
        # Adjust FC input size for bidirectional LSTM
        lstm_output_dim = hidden_dim * 2  # *2 for bidirectional
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass"""
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Concatenate forward and backward final hidden states
        combined = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        return self.fc(combined)
