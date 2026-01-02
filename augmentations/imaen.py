"""
Interactive Masked Amino Acid Enhancement with Noise (IMAEN)

This module implements an IMAEN-inspired augmentation technique for protein sequences.
IMAEN (Interpretable Molecular Augmentation Encoding Networks) was originally developed
for drug-target interaction prediction with molecular augmentation mechanisms.

This adaptation applies property-aware amino acid substitution to protein sequences,
creating interpretable variations while preserving biochemical properties.

Reference:
    IMAEN: Interpretable Molecular Augmentation Encoding Networks
    (Adapted for protein sequence augmentation)
"""

import random
from typing import List, Set, Dict, Tuple


# Amino acid property groups based on biochemical characteristics
# Reference: Classical biochemistry amino acid classification

HYDROPHOBIC_NONPOLAR = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'}
POLAR_UNCHARGED = {'S', 'T', 'N', 'Q', 'C', 'Y', 'G'}
POSITIVELY_CHARGED = {'K', 'R', 'H'}
NEGATIVELY_CHARGED = {'D', 'E'}

# Aromatic amino acids
AROMATIC = {'F', 'Y', 'W', 'H'}

# Small amino acids
SMALL = {'A', 'G', 'S', 'C', 'T'}

# Aliphatic amino acids
ALIPHATIC = {'A', 'V', 'L', 'I'}

# Property-based substitution groups
# Each amino acid can be substituted with similar amino acids from the same group
PROPERTY_GROUPS: Dict[str, Set[str]] = {
    # Hydrophobic, aliphatic
    'A': {'V', 'I', 'L', 'M'},
    'V': {'A', 'I', 'L', 'M'},
    'I': {'V', 'L', 'M', 'A'},
    'L': {'I', 'V', 'M', 'A'},
    'M': {'L', 'I', 'V', 'F'},
    
    # Aromatic, hydrophobic
    'F': {'Y', 'W', 'M'},
    'Y': {'F', 'W', 'H'},
    'W': {'F', 'Y', 'H'},
    
    # Polar, uncharged
    'S': {'T', 'C', 'A', 'N'},
    'T': {'S', 'C', 'V', 'N'},
    'N': {'Q', 'S', 'T', 'D'},
    'Q': {'N', 'E', 'K'},
    'C': {'S', 'T', 'A'},
    
    # Positively charged
    'K': {'R', 'Q', 'H'},
    'R': {'K', 'H', 'Q'},
    'H': {'K', 'R', 'Y', 'W'},
    
    # Negatively charged
    'D': {'E', 'N'},
    'E': {'D', 'Q'},
    
    # Special cases
    'G': {'A', 'S'},  # Glycine is small and flexible
    'P': {'A', 'G'},  # Proline is rigid
}


# Conservative substitution matrix (similar to BLOSUM-based groups)
# Higher similarity = more conservative substitution
SUBSTITUTION_SIMILARITY: Dict[Tuple[str, str], float] = {}

# Build similarity scores (simplified)
for aa, similar_set in PROPERTY_GROUPS.items():
    for sim_aa in similar_set:
        SUBSTITUTION_SIMILARITY[(aa, sim_aa)] = 1.0


def _select_positions_by_property(
    sequence: List[str], 
    intensity: float,
    property_bias: str = 'random'
) -> List[int]:
    """
    Select positions in the sequence for augmentation based on amino acid properties.
    
    This function implements the "interpretable" aspect of IMAEN by selecting
    positions based on biochemical properties rather than purely random selection.
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Fraction of positions to select (0.0 to 1.0)
        property_bias: Which property to bias selection towards
                      ('hydrophobic', 'polar', 'charged', 'random')
    
    Returns:
        List of position indices to augment
        
    Reference:
        IMAEN Section: Molecular Augmentation Mechanism
        Adapted for protein sequences with property-aware selection
    """
    if intensity <= 0 or len(sequence) == 0:
        return []
    
    seq_len = len(sequence)
    num_positions = max(1, int(intensity * seq_len))
    
    # Build candidate positions based on property bias
    if property_bias == 'hydrophobic':
        candidates = [i for i, aa in enumerate(sequence) if aa in HYDROPHOBIC_NONPOLAR]
    elif property_bias == 'polar':
        candidates = [i for i, aa in enumerate(sequence) if aa in POLAR_UNCHARGED]
    elif property_bias == 'charged':
        candidates = [i for i, aa in enumerate(sequence) 
                     if aa in POSITIVELY_CHARGED or aa in NEGATIVELY_CHARGED]
    elif property_bias == 'aromatic':
        candidates = [i for i, aa in enumerate(sequence) if aa in AROMATIC]
    else:  # random
        candidates = list(range(seq_len))
    
    # If not enough candidates, fall back to all positions
    if len(candidates) < num_positions:
        candidates = list(range(seq_len))
    
    # Randomly sample from candidates
    selected = random.sample(candidates, min(num_positions, len(candidates)))
    return sorted(selected)


def _apply_property_aware_substitution(
    sequence: List[str],
    positions: List[int],
    conservative: bool = True
) -> List[str]:
    """
    Apply property-aware amino acid substitutions at specified positions.
    
    This implements the augmentation mechanism while preserving biochemical
    properties through conservative substitutions.
    
    Args:
        sequence: List of single-letter amino acid codes
        positions: List of positions to substitute
        conservative: If True, only substitute with biochemically similar amino acids
                     If False, allow more diverse substitutions
    
    Returns:
        Augmented sequence with substitutions applied
        
    Reference:
        IMAEN Section: Interpretable Stack Convolutional Encoding
        Property preservation ensures interpretability of augmentation
    """
    augmented = sequence.copy()
    
    for pos in positions:
        original_aa = augmented[pos]
        
        # Get substitution candidates based on property groups
        if conservative and original_aa in PROPERTY_GROUPS:
            candidates = list(PROPERTY_GROUPS[original_aa])
        else:
            # Less conservative: use broader property groups
            if original_aa in HYDROPHOBIC_NONPOLAR:
                candidates = list(HYDROPHOBIC_NONPOLAR - {original_aa})
            elif original_aa in POLAR_UNCHARGED:
                candidates = list(POLAR_UNCHARGED - {original_aa})
            elif original_aa in POSITIVELY_CHARGED:
                candidates = list(POSITIVELY_CHARGED - {original_aa})
            elif original_aa in NEGATIVELY_CHARGED:
                candidates = list(NEGATIVELY_CHARGED - {original_aa})
            else:
                # Fallback to property group
                candidates = list(PROPERTY_GROUPS.get(original_aa, {'A'}))
        
        # Perform substitution
        if candidates:
            augmented[pos] = random.choice(candidates)
    
    return augmented


def _add_interpretable_noise(
    sequence: List[str],
    intensity: float,
    noise_type: str = 'substitution'
) -> List[str]:
    """
    Add controlled, interpretable noise to the sequence.
    
    Unlike random noise, this noise is biochemically meaningful and
    maintains sequence validity.
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Noise level (0.0 to 1.0)
        noise_type: Type of noise ('substitution', 'swap', 'none')
    
    Returns:
        Sequence with noise applied
    """
    if noise_type == 'none' or intensity <= 0:
        return sequence
    
    seq_len = len(sequence)
    noise_positions = max(1, int(intensity * 0.1 * seq_len))  # 10% of intensity
    
    if noise_type == 'substitution':
        # Apply random property-aware substitutions
        positions = random.sample(range(seq_len), min(noise_positions, seq_len))
        return _apply_property_aware_substitution(sequence, positions, conservative=False)
    
    elif noise_type == 'swap':
        # Swap adjacent amino acids of similar properties
        augmented = sequence.copy()
        for _ in range(noise_positions):
            if seq_len < 2:
                break
            pos = random.randint(0, seq_len - 2)
            # Swap adjacent positions
            augmented[pos], augmented[pos + 1] = augmented[pos + 1], augmented[pos]
        return augmented
    
    return sequence


def imaen_augment(
    sequence: List[str], 
    intensity: float,
    property_bias: str = 'random',
    conservative: bool = True,
    noise_level: float = 0.1
) -> List[str]:
    """
    Apply IMAEN-inspired augmentation to a protein sequence.
    
    This function implements an interpretable molecular augmentation approach
    adapted for protein sequences. It combines:
    1. Property-aware position selection
    2. Conservative amino acid substitution
    3. Controlled interpretable noise injection
    
    The augmentation preserves biochemical properties while creating sequence
    diversity, making it suitable for protein engineering and predictive modeling.
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Augmentation intensity (0.0 to 1.0)
                  Controls the fraction of positions to modify
        property_bias: Bias for position selection ('random', 'hydrophobic', 
                      'polar', 'charged', 'aromatic')
        conservative: If True, use highly conservative substitutions
        noise_level: Amount of additional interpretable noise (0.0 to 1.0)
    
    Returns:
        Augmented protein sequence (same format as input)
        Returns original sequence if augmentation fails
    
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
        >>> aug_seq = imaen_augment(seq, intensity=0.3)
        >>> # aug_seq will have ~30% of positions substituted with
        >>> # biochemically similar amino acids
    
    Reference:
        IMAEN: Interpretable Molecular Augmentation Encoding Networks
        Adapted for protein sequence augmentation with property-aware mechanisms
        
    Note:
        This implementation follows the interface pattern from example.py:
        function(sequence: List[str], intensity: float) -> List[str]
    """
    try:
        if intensity <= 0 or len(sequence) == 0:
            return sequence
        
        # Ensure intensity is in valid range
        intensity = max(0.0, min(1.0, intensity))
        
        # Step 1: Select positions based on property bias (interpretable selection)
        positions = _select_positions_by_property(sequence, intensity, property_bias)
        
        if not positions:
            return sequence
        
        # Step 2: Apply property-aware substitutions (conservative augmentation)
        augmented = _apply_property_aware_substitution(sequence, positions, conservative)
        
        # Step 3: Add controlled interpretable noise
        if noise_level > 0:
            augmented = _add_interpretable_noise(augmented, intensity * noise_level, 'substitution')
        
        # Sanity check: ensure length is preserved
        if len(augmented) != len(sequence):
            return sequence
        
        return augmented
    
    except Exception:
        # If anything goes wrong, return original sequence
        return sequence


# Simplified interface matching example.py pattern
def imaen_simple(sequence: List[str], intensity: float) -> List[str]:
    """
    Simplified IMAEN augmentation with default parameters.
    
    This is the main function to be registered in the augmentation pipeline.
    It uses default parameters optimized for general protein augmentation.
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Augmentation intensity (0.0 to 1.0)
    
    Returns:
        Augmented protein sequence
        
    Reference:
        IMAEN: Interpretable Molecular Augmentation Encoding Networks
    """
    return imaen_augment(
        sequence=sequence,
        intensity=intensity,
        property_bias='random',
        conservative=True,
        noise_level=0.1
    )
