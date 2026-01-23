"""
Protein augmentation techniques.

This module contains various augmentation strategies for protein sequences,
including nucleotide-level and masking augmentations.
"""

from .nta_augmentation import nucleotide_augment
from .residue_masking import mask_residues, simple_mask_residues, conservative_mask_residues
from .imaen import imaen_simple, imaen_augment

__all__ = [
    'nucleotide_augment',
    'mask_residues',
    'simple_mask_residues',
    'conservative_mask_residues',
    'imaen_simple',
    'imaen_augment'
]
