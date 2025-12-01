"""
Residue Masking Augmentation for Protein Sequences

This module implements residue masking augmentation techniques inspired by
masked language modeling (MLM) approaches used in protein language models
such as ProtBERT, ESM-1b, and ESM-2.

The technique involves randomly masking (replacing) amino acid residues with:
- A special mask token ('X' for unknown amino acid)
- Random amino acids
- Or leaving them unchanged

This creates a denoising-style augmentation that encourages models to learn
robust representations despite incomplete or noisy sequence information.

References:
- BERT: Devlin et al. 2019, "BERT: Pre-training of Deep Bidirectional Transformers"
- ProtBERT: Elnaggar et al. 2020, "ProtTrans: Towards Cracking the Language of Life"
- ESM: Rives et al. 2021, "Biological structure and function emerge from scaling unsupervised learning"
"""

import random
from typing import List


# Standard 20 amino acids
AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


def mask_residues(
    sequence: List[str],
    masking_rate: float,
    mask_token: str = 'X',
    random_rate: float = 0.1,
    unchanged_rate: float = 0.1
) -> List[str]:
    """
    Apply MLM-style residue masking to a protein sequence.
    
    This function implements the masking strategy used in BERT and adapted
    for proteins in ProtBERT and ESM models. For each selected position:
    - With probability (1 - random_rate - unchanged_rate): Replace with mask_token
    - With probability random_rate: Replace with random amino acid
    - With probability unchanged_rate: Leave unchanged
    
    The default parameters (80% mask, 10% random, 10% unchanged) follow
    the original BERT masking strategy.
    
    Args:
        sequence: List of single-letter amino acid codes
        masking_rate: Fraction of residues to select for masking (0.0 to 1.0)
        mask_token: Token to use for masking (default: 'X' for unknown)
        random_rate: Probability of replacing with random AA (default: 0.1)
        unchanged_rate: Probability of leaving unchanged (default: 0.1)
        
    Returns:
        Masked amino acid sequence (same format as input)
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'Y', 'L', 'A', 'M', 'I', 'N', 'E']
        >>> masked = mask_residues(seq, 0.15)
        >>> # Approximately 15% of positions will be masked/modified
        
    Reference:
        Elnaggar, A. et al. (2020)
        "ProtTrans: Towards Cracking the Language of Life's Code Through 
         Self-Supervised Deep Learning and High Performance Computing"
        https://doi.org/10.1101/2020.07.12.199554
        
    Note:
        This implementation is compatible with the augmentation interface used
        in example.py, where augmentation functions take (sequence, intensity)
        and return an augmented sequence.
    """
    if masking_rate <= 0:
        return sequence
    
    # Create a copy to avoid modifying original sequence
    masked_sequence = sequence.copy()
    seq_len = len(masked_sequence)
    
    # Calculate number of positions to mask
    num_to_mask = max(1, int(masking_rate * seq_len))
    
    # Randomly select positions to mask
    positions_to_mask = random.sample(range(seq_len), min(num_to_mask, seq_len))
    
    # Calculate probabilities
    # mask_prob = 1 - random_rate - unchanged_rate (typically 0.8)
    mask_prob = max(0.0, 1.0 - random_rate - unchanged_rate)
    
    # Apply masking strategy
    for pos in positions_to_mask:
        rand_val = random.random()
        
        if rand_val < mask_prob:
            # Replace with mask token (80% by default)
            masked_sequence[pos] = mask_token
        elif rand_val < mask_prob + random_rate:
            # Replace with random amino acid (10% by default)
            masked_sequence[pos] = random.choice(AMINO_ACIDS)
        # else: leave unchanged (10% by default)
    
    return masked_sequence


def simple_mask_residues(
    sequence: List[str],
    masking_rate: float,
    mask_token: str = 'X'
) -> List[str]:
    """
    Simple residue masking: selected positions are replaced with mask token.
    
    This is a simplified version of mask_residues that always replaces
    selected positions with the mask token (no random or unchanged).
    Useful for basic masking without the complexity of MLM-style masking.
    
    Args:
        sequence: List of single-letter amino acid codes
        masking_rate: Fraction of residues to mask (0.0 to 1.0)
        mask_token: Token to use for masking (default: 'X' for unknown)
        
    Returns:
        Masked amino acid sequence (same format as input)
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H']
        >>> masked = simple_mask_residues(seq, 0.5)
        >>> # Exactly 50% of positions will be 'X'
        
    Reference:
        Simplified variant of the masking strategy described in:
        Devlin, J. et al. (2019) "BERT: Pre-training of Deep Bidirectional 
        Transformers for Language Understanding"
    """
    if masking_rate <= 0:
        return sequence
    
    # Create a copy to avoid modifying original sequence
    masked_sequence = sequence.copy()
    seq_len = len(masked_sequence)
    
    # Calculate number of positions to mask
    num_to_mask = max(1, int(masking_rate * seq_len))
    
    # Randomly select positions to mask
    positions_to_mask = random.sample(range(seq_len), min(num_to_mask, seq_len))
    
    # Replace selected positions with mask token
    for pos in positions_to_mask:
        masked_sequence[pos] = mask_token
    
    return masked_sequence


def conservative_mask_residues(
    sequence: List[str],
    masking_rate: float,
    mask_token: str = 'X'
) -> List[str]:
    """
    Conservative masking: preserve chemically similar amino acids.
    
    This variant masks residues but tries to preserve chemical properties
    by masking within groups of similar amino acids.
    
    Groups:
    - Hydrophobic: A, V, I, L, M, F, W, P
    - Hydrophilic: S, T, N, Q
    - Charged positive: K, R, H
    - Charged negative: D, E
    - Special: C, G, Y
    
    Args:
        sequence: List of single-letter amino acid codes
        masking_rate: Fraction of residues to mask (0.0 to 1.0)
        mask_token: Token to use when no suitable replacement found
        
    Returns:
        Conservatively masked amino acid sequence
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H']
        >>> masked = conservative_mask_residues(seq, 0.5)
        >>> # M might be replaced with L (both hydrophobic)
        >>> # E might be replaced with D (both negative charged)
    """
    if masking_rate <= 0:
        return sequence
    
    # Define amino acid groups by chemical properties
    aa_groups = {
        'A': ['V', 'I', 'L', 'M', 'F', 'W', 'P'],  # Hydrophobic
        'V': ['A', 'I', 'L', 'M'],
        'I': ['A', 'V', 'L', 'M'],
        'L': ['A', 'V', 'I', 'M'],
        'M': ['A', 'V', 'I', 'L'],
        'F': ['W', 'Y'],
        'W': ['F', 'Y'],
        'P': ['A'],
        'S': ['T', 'N', 'Q'],  # Hydrophilic
        'T': ['S', 'N', 'Q'],
        'N': ['S', 'T', 'Q'],
        'Q': ['S', 'T', 'N'],
        'K': ['R', 'H'],  # Positive
        'R': ['K', 'H'],
        'H': ['K', 'R'],
        'D': ['E'],  # Negative
        'E': ['D'],
        'C': ['C'],  # Special (Cysteine usually conserved)
        'G': ['G'],  # Special (Glycine unique)
        'Y': ['F', 'W'],  # Aromatic
    }
    
    # Create a copy
    masked_sequence = sequence.copy()
    seq_len = len(masked_sequence)
    
    # Calculate number of positions to mask
    num_to_mask = max(1, int(masking_rate * seq_len))
    
    # Randomly select positions to mask
    positions_to_mask = random.sample(range(seq_len), min(num_to_mask, seq_len))
    
    # Apply conservative masking
    for pos in positions_to_mask:
        original_aa = masked_sequence[pos]
        
        if original_aa in aa_groups and aa_groups[original_aa]:
            # Replace with chemically similar amino acid
            masked_sequence[pos] = random.choice(aa_groups[original_aa])
        else:
            # Fall back to mask token if no group found
            masked_sequence[pos] = mask_token
    
    return masked_sequence
