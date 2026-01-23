"""
Spider Augmentation for Protein Sequences

This module implements the Spider augmentation technique as described in:
"A Deep Learning Approach with Data Augmentation to Predict 
Novel Spider Neurotoxic Peptides"

Reference:
    The Spider augmentation method employs two primary mechanisms:
    1. Random substitution of amino acids
    2. Random insertion of amino acids
    
    Original paper includes BLAST-based homology filtering for biological
    plausibility. This implementation provides a simplified version focusing
    on the core augmentation operations for compatibility with the framework.

Authors: Implementation based on the Spider neurotoxic peptide paper
"""

import random
from typing import List


# Standard 20 amino acids
AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


def random_substitute_residues(
    sequence: List[str],
    substitution_rate: float
) -> List[str]:
    """
    Randomly substitute amino acids in a protein sequence.
    
    This function implements the substitution component of Spider augmentation.
    Selected positions are replaced with random amino acids from the standard 20.
    
    Args:
        sequence: List of single-letter amino acid codes
        substitution_rate: Proportion of residues to substitute (0.0 to 1.0)
        
    Returns:
        List of amino acids with random substitutions applied
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'Y', 'L']
        >>> result = random_substitute_residues(seq, 0.5)
        >>> # Approximately 3 residues will be substituted with random AAs
        
    Reference:
        Spider paper, Section on Data Augmentation: Random amino acid substitution
    """
    if substitution_rate <= 0:
        return sequence.copy()
    
    # Create a copy to avoid modifying the original
    augmented = sequence.copy()
    seq_len = len(augmented)
    
    if seq_len == 0:
        return augmented
    
    # Calculate number of positions to substitute
    num_substitutions = max(0, int(substitution_rate * seq_len))
    
    if num_substitutions == 0:
        return augmented
    
    # Randomly select positions to substitute
    positions = random.sample(range(seq_len), min(num_substitutions, seq_len))
    
    # Substitute with random amino acids
    for pos in positions:
        augmented[pos] = random.choice(AMINO_ACIDS)
    
    return augmented


def random_insert_residues(
    sequence: List[str],
    insertion_rate: float
) -> List[str]:
    """
    Randomly insert amino acids into a protein sequence.
    
    This function implements the insertion component of Spider augmentation.
    Random amino acids are inserted at random positions throughout the sequence.
    
    Args:
        sequence: List of single-letter amino acid codes
        insertion_rate: Proportion of sequence length to insert (0.0 to 1.0)
                       e.g., 0.3 means insert 30% of original length
        
    Returns:
        List of amino acids with random insertions applied
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'Y', 'L']  # Length 6
        >>> result = random_insert_residues(seq, 0.3)
        >>> # Approximately 2 new residues inserted (6 * 0.3 ≈ 2)
        >>> # Result length ≈ 8
        
    Reference:
        Spider paper, Section on Data Augmentation: Random amino acid insertion
    """
    if insertion_rate <= 0:
        return sequence.copy()
    
    # Create a copy to avoid modifying the original
    augmented = sequence.copy()
    seq_len = len(augmented)
    
    if seq_len == 0:
        return augmented
    
    # Calculate number of residues to insert
    num_insertions = max(0, int(insertion_rate * seq_len))
    
    if num_insertions == 0:
        return augmented
    
    # Perform insertions
    # Insert from end to beginning to avoid index shifting issues
    for _ in range(num_insertions):
        # Random position (can be at the end)
        pos = random.randint(0, len(augmented))
        # Random amino acid
        aa = random.choice(AMINO_ACIDS)
        # Insert
        augmented.insert(pos, aa)
    
    return augmented


def spider_augment(
    sequence: List[str],
    intensity: float
) -> List[str]:
    """
    Apply Spider augmentation to a protein sequence.
    
    This function implements the Spider augmentation technique, which combines
    random amino acid substitution and insertion to create diverse protein
    sequences. The technique was originally developed for augmenting neurotoxic
    peptide datasets.
    
    The augmentation process:
    1. Apply random amino acid substitutions
    2. Apply random amino acid insertions
    
    The intensity parameter controls both operations:
    - Substitution rate = intensity * 0.5
    - Insertion rate = intensity * 0.5
    
    This balanced split allows both operations to contribute to sequence diversity
    while keeping the total modification proportional to the intensity.
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Augmentation intensity (0.0 to 1.0)
                  0.0 = no augmentation
                  1.0 = maximum augmentation (50% substitution + 50% insertion)
        
    Returns:
        Augmented protein sequence as a list of amino acid codes
        
    Example:
        >>> original = ['M', 'E', 'T', 'H', 'Y', 'L', 'A', 'M', 'I', 'N', 'E']
        >>> augmented = spider_augment(original, 0.3)
        >>> # Some residues substituted, some new residues inserted
        >>> # Length may increase due to insertions
        
    Integration:
        This function follows the framework interface pattern:
        func(sequence: List[str], intensity: float) -> List[str]
        
        Used in example.py as part of AUGMENTATION_FUNCTIONS list.
        
    Reference:
        Paper: "A Deep Learning Approach with Data Augmentation to Predict
               Novel Spider Neurotoxic Peptides"
        
        Note: Original paper includes BLAST-based homology filtering (E-value 
        threshold 1×10⁻⁵) to select biologically plausible sequences. This 
        implementation provides the core augmentation operations without the 
        filtering step for framework simplicity and speed.
    """
    if intensity <= 0:
        return sequence.copy()
    
    # Clamp intensity to valid range
    intensity = min(1.0, max(0.0, intensity))
    
    # Split intensity between substitution and insertion
    # Using 50/50 split for balanced augmentation
    substitution_rate = intensity * 0.5
    insertion_rate = intensity * 0.5
    
    # Apply substitutions first
    augmented = random_substitute_residues(sequence, substitution_rate)
    
    # Then apply insertions
    augmented = random_insert_residues(augmented, insertion_rate)
    
    return augmented
