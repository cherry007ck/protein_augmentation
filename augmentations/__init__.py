"""
Protein sequence augmentation modules.

This package contains various augmentation techniques for protein sequences:
- NTA (Nucleotide Augmentation): Back-translation with synonymous substitutions
- Residue Masking: MLM-style masking for denoising augmentation
- RSA (Retrieved Sequence Augmentation): Conservative mutations simulating homologs
"""

from .nta_augmentation import nucleotide_augment
from .residue_masking import mask_residues, conservative_mask_residues, simple_mask_residues
from .rsa_augmentation import rsa_augment, rsa_augment_with_original

__all__ = [
    'nucleotide_augment',
    'mask_residues',
    'conservative_mask_residues',
    'simple_mask_residues',
    'rsa_augment',
    'rsa_augment_with_original',
]
