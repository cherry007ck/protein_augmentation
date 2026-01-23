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
This module contains various augmentation strategies for protein sequences,
including nucleotide-level, masking, and semantic augmentations.
"""

from .nta_augmentation import nucleotide_augment
from .residue_masking import mask_residues, simple_mask_residues, conservative_mask_residues
from .nana_augmentation import nana_augment
from .migu_augmentation import migu_augment
from .imaen import imaen_simple, imaen_augment

__all__ = [
    'nucleotide_augment',
    'mask_residues',
    'preis_augment',
    'simple_mask_residues',
    'conservative_mask_residues',
    'nana_augment',
    'migu_augment',
    'imaen_simple',
    'imaen_augment'
]
