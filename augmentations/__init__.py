"""
Protein Sequence Augmentation Techniques

This package contains various augmentation methods for protein sequences,
designed to work with the PyTorch-based protein data framework.

Available augmentation techniques:
- NTA (Nucleotide Augmentation): Synonymous codon substitution
- Residue Masking: MLM-style and conservative masking
- PreIS (Supervised Data Augmentation): Self-mixing with segment swapping and token shuffling
"""

from .nta_augmentation import nucleotide_augment
from .residue_masking import mask_residues, conservative_mask_residues
from .preis_augmentation import preis_augment

__all__ = [
    'nucleotide_augment',
    'mask_residues',
    'conservative_mask_residues',
    'preis_augment',
]
