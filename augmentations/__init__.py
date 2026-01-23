"""
Protein augmentation techniques.

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
    'simple_mask_residues',
    'conservative_mask_residues',
    'nana_augment',
    'migu_augment',
    'imaen_simple',
    'imaen_augment'
]
