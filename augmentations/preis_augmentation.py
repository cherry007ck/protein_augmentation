"""
PreIS Augmentation for Protein Sequences (Simplified Self-Mixing Version)

This module implements a simplified version of the PreIS (Predictive Edit-based 
Interpolation and Substitution) data augmentation technique described in:
"PreIS: A Novel Data Augmentation Approach Using Protein Language Models 
for Influenza A Subtype Prediction"

The original PreIS/SDA (Supervised Data Augmentation) performs label-aware 
augmentation by mixing segments and tokens from sequences with the same label.
This simplified version performs self-mixing within a single sequence to maintain
compatibility with the existing framework's functional interface.

Augmentation Strategy:
1. Global Segment Swapping: Swap two random segments within the sequence
2. Local Token Shuffling: Shuffle random individual tokens within the sequence

This approach preserves:
- Sequence length
- Amino acid composition (multiset)
- Overall sequence characteristics

while introducing controlled diversity through structured permutation.

References:
- PreIS/SDA original implementation: paper_codes/PreIS/PreIS/utils.py
- Default parameters from config.py: GAMMA_G=0.4, GAMMA_L=0.1
"""

import random
from typing import List, Tuple


def preis_global_mixing(sequence: List[str], gamma_g: float) -> List[str]:
    """
    Apply global segment swapping within a sequence.
    
    Selects two random segments of specified length and swaps them.
    This is inspired by the global segment replacement in the original
    PreIS/SDA, but adapted to work within a single sequence.
    
    Args:
        sequence: List of single-letter amino acid codes
        gamma_g: Fraction of sequence to use as segment length (0.0 to 1.0)
                Typically 0.4 in the original PreIS implementation
        
    Returns:
        Sequence with two segments swapped
        
    Reference:
        PreIS paper, Section on Supervised Data Augmentation (SDA)
        Original GAMMA_G parameter: 0.4
    """
    if gamma_g <= 0 or len(sequence) < 4:
        return sequence
    
    # Create a copy to avoid modifying the original
    seq = sequence.copy()
    seq_len = len(seq)
    
    # Calculate segment length
    segment_len = max(1, int(gamma_g * seq_len))
    # Ensure we can fit two segments
    segment_len = min(segment_len, seq_len // 2)
    
    if segment_len < 1:
        return seq
    
    # Select two random non-overlapping segments
    max_start = seq_len - segment_len
    if max_start < 1:
        return seq
    
    # First segment
    start1 = random.randint(0, max_start)
    end1 = start1 + segment_len
    
    # Second segment (ensure non-overlapping)
    attempts = 0
    while attempts < 10:
        start2 = random.randint(0, max_start)
        end2 = start2 + segment_len
        
        # Check if segments don't overlap
        if end1 <= start2 or end2 <= start1:
            break
        attempts += 1
    
    # If we found non-overlapping segments, swap them
    if attempts < 10:
        # Extract segments
        segment1 = seq[start1:end1]
        segment2 = seq[start2:end2]
        
        # Swap segments
        seq[start1:end1] = segment2
        seq[start2:end2] = segment1
    
    return seq


def preis_local_mixing(sequence: List[str], gamma_l: float) -> List[str]:
    """
    Apply local token shuffling within a sequence.
    
    Randomly shuffles individual tokens (amino acids) at random positions.
    This is inspired by the local token replacement in the original PreIS/SDA,
    but adapted to shuffle tokens within the same sequence.
    
    Args:
        sequence: List of single-letter amino acid codes
        gamma_l: Fraction of sequence to shuffle (0.0 to 1.0)
                Typically 0.1 in the original PreIS implementation
        
    Returns:
        Sequence with random tokens shuffled
        
    Reference:
        PreIS paper, Section on Supervised Data Augmentation (SDA)
        Original GAMMA_L parameter: 0.1
    """
    if gamma_l <= 0 or len(sequence) < 2:
        return sequence
    
    # Create a copy
    seq = sequence.copy()
    seq_len = len(seq)
    
    # Calculate number of tokens to shuffle
    num_tokens = max(0, int(gamma_l * seq_len))
    
    if num_tokens < 2:
        return seq
    
    # Select random positions to shuffle
    num_tokens = min(num_tokens, seq_len)
    positions = random.sample(range(seq_len), num_tokens)
    
    # Extract values at these positions
    values = [seq[pos] for pos in positions]
    
    # Shuffle the values
    random.shuffle(values)
    
    # Put shuffled values back
    for pos, val in zip(positions, values):
        seq[pos] = val
    
    return seq


def preis_augment(sequence: List[str], intensity: float) -> List[str]:
    """
    Apply PreIS-inspired augmentation (simplified self-mixing version).
    
    This function implements a simplified version of the PreIS/SDA technique
    that performs both global segment swapping and local token shuffling
    within the same sequence to create diversity while preserving overall
    sequence composition.
    
    The augmentation applies:
    1. Global segment swapping (controlled by GAMMA_G)
    2. Local token shuffling (controlled by GAMMA_L)
    
    Both operations preserve:
    - Sequence length
    - Amino acid composition (multiset of residues)
    - Overall sequence characteristics
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Controls augmentation strength (0.0 to 1.0+)
                  This parameter scales both GAMMA_G and GAMMA_L.
                  Typical usage: 0.1 (light), 0.5 (moderate), 1.0 (strong)
                  
    Returns:
        Augmented amino acid sequence (same format as input)
        Returns original sequence if augmentation fails or sequence is too short
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
        >>> aug_seq = preis_augment(seq, 0.5)
        >>> # aug_seq will have segments swapped and tokens shuffled
        >>> # but will contain the same amino acids in different order
        
    Reference:
        Original PreIS/SDA paper: "PreIS: A Novel Data Augmentation Approach 
        Using Protein Language Models for Influenza A Subtype Prediction"
        
        Original implementation: paper_codes/PreIS/PreIS/utils.py (SDA function)
        
        This simplified version performs self-mixing instead of cross-sequence
        mixing to maintain compatibility with the framework's functional interface.
        
        Default parameters:
        - GAMMA_G = 0.4 (40% of sequence for global segment)
        - GAMMA_L = 0.1 (10% of sequence for local shuffling)
        
    Note:
        This implementation is compatible with the augmentation interface used
        in example.py, where augmentation functions take (sequence, intensity)
        and return an augmented sequence.
        
        For the original label-aware cross-sequence mixing behavior, a dataset-aware
        interface would be required (future enhancement).
    """
    try:
        # Validate input
        if not sequence or len(sequence) < 4:
            return sequence
        
        if intensity <= 0:
            return sequence
        
        # Scale parameters by intensity
        # Default parameters from PreIS config: GAMMA_G=0.4, GAMMA_L=0.1
        gamma_g = 0.4 * intensity
        gamma_l = 0.1 * intensity
        
        # Clamp to reasonable values
        gamma_g = min(gamma_g, 0.5)  # Max 50% segment length
        gamma_l = min(gamma_l, 0.3)  # Max 30% token shuffling
        
        # Apply global segment swapping
        augmented = preis_global_mixing(sequence, gamma_g)
        
        # Apply local token shuffling
        augmented = preis_local_mixing(augmented, gamma_l)
        
        # Sanity check: ensure length is preserved
        if len(augmented) != len(sequence):
            return sequence
        
        return augmented
    
    except Exception:
        # If anything goes wrong, return original sequence
        return sequence
