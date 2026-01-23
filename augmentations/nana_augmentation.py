"""
NaNa (Novel Augmentation of New Node Attributes) for Protein Sequences

This module implements NaNa augmentation as described in:
"NaNa and MiGu: Semantic Data Augmentation Techniques to Enhance Protein 
Classification in Graph Neural Networks"

NaNa enriches protein representations by incorporating:
- Molecular biophysical features (hydrophobicity, charge, pKa)
- Secondary structure propensities (α-helix, β-sheet, turn)
- Amino acid type features

This implementation adapts the graph-based NaNa technique to sequence-based
augmentation, making it compatible with the existing framework interface.

Reference:
    Paper: "NaNa and MiGu: Semantic Data Augmentation Techniques..."
    arXiv: https://arxiv.org/
"""

import random
from typing import List, Dict, Tuple


# Standard 20 amino acids
AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


# Kyte-Doolittle Hydrophobicity Scale
# More positive = more hydrophobic, More negative = more hydrophilic
HYDROPHOBICITY: Dict[str, float] = {
    'A': 1.8,   # Alanine
    'C': 2.5,   # Cysteine
    'D': -3.5,  # Aspartic acid
    'E': -3.5,  # Glutamic acid
    'F': 2.8,   # Phenylalanine
    'G': -0.4,  # Glycine
    'H': -3.2,  # Histidine
    'I': 4.5,   # Isoleucine
    'K': -3.9,  # Lysine
    'L': 3.8,   # Leucine
    'M': 1.9,   # Methionine
    'N': -3.5,  # Asparagine
    'P': -1.6,  # Proline
    'Q': -3.5,  # Glutamine
    'R': -4.5,  # Arginine
    'S': -0.8,  # Serine
    'T': -0.7,  # Threonine
    'V': 4.2,   # Valine
    'W': -0.9,  # Tryptophan
    'Y': -1.3,  # Tyrosine
}


# Charge at pH 7.4
CHARGE: Dict[str, int] = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}


# Van der Waals Volume (Å³)
SIZE: Dict[str, float] = {
    'A': 67,   'C': 86,   'D': 91,   'E': 109,  'F': 135,
    'G': 48,   'H': 118,  'I': 124,  'K': 135,  'L': 124,
    'M': 124,  'N': 96,   'P': 90,   'Q': 114,  'R': 148,
    'S': 73,   'T': 93,   'V': 105,  'W': 163,  'Y': 141,
}


# Chou-Fasman Secondary Structure Propensities
# α-helix propensity
HELIX_PROPENSITY: Dict[str, float] = {
    'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
    'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
    'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
    'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69,
}


# β-sheet propensity
SHEET_PROPENSITY: Dict[str, float] = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
}


# Turn propensity
TURN_PROPENSITY: Dict[str, float] = {
    'A': 0.66, 'C': 1.19, 'D': 1.46, 'E': 0.74, 'F': 0.60,
    'G': 1.56, 'H': 0.95, 'I': 0.47, 'K': 1.01, 'L': 0.59,
    'M': 0.60, 'N': 1.56, 'P': 1.52, 'Q': 0.98, 'R': 0.95,
    'S': 1.43, 'T': 0.96, 'V': 0.50, 'W': 0.96, 'Y': 1.14,
}


# Amino acid groups based on combined biophysical properties
# These groups allow conservative substitutions within similar amino acids
BIOPHYSICAL_GROUPS: Dict[str, List[str]] = {
    # Hydrophobic aliphatic
    'A': ['V', 'I', 'L'],
    'V': ['A', 'I', 'L'],
    'I': ['V', 'L', 'M'],
    'L': ['I', 'V', 'M'],
    'M': ['I', 'L'],
    
    # Aromatic
    'F': ['W', 'Y'],
    'W': ['F', 'Y'],
    'Y': ['F', 'W'],
    
    # Polar uncharged
    'S': ['T', 'N', 'Q'],
    'T': ['S', 'N', 'Q'],
    'N': ['S', 'T', 'Q'],
    'Q': ['N', 'T', 'S'],
    
    # Positively charged
    'K': ['R', 'H'],
    'R': ['K', 'H'],
    'H': ['K', 'R'],
    
    # Negatively charged
    'D': ['E'],
    'E': ['D'],
    
    # Special cases
    'C': ['S'],  # Cysteine can sometimes substitute with serine
    'G': ['A'],  # Glycine is unique, but alanine is smallest alternative
    'P': ['A'],  # Proline is unique, alanine in rare cases
}


def calculate_similarity(aa1: str, aa2: str) -> float:
    """
    Calculate multi-dimensional similarity between two amino acids.
    
    Combines biophysical properties (hydrophobicity, charge, size) and
    secondary structure propensities to compute a similarity score.
    
    Args:
        aa1: First amino acid (single letter code)
        aa2: Second amino acid (single letter code)
        
    Returns:
        Similarity score (0.0 to 1.0, higher = more similar)
        
    Reference:
        Based on NaNa's incorporation of molecular biophysics and
        secondary structure properties for semantic augmentation.
    """
    if aa1 not in AMINO_ACIDS or aa2 not in AMINO_ACIDS:
        return 0.0
    
    if aa1 == aa2:
        return 1.0
    
    # Normalize differences to [0, 1] range
    # Hydrophobicity difference (range: -4.5 to 4.5, max diff ≈ 9)
    hydro_diff = abs(HYDROPHOBICITY[aa1] - HYDROPHOBICITY[aa2]) / 9.0
    
    # Charge difference (range: -1 to 1, max diff = 2)
    charge_diff = abs(CHARGE[aa1] - CHARGE[aa2]) / 2.0
    
    # Size difference (range: 48 to 163, max diff ≈ 115)
    size_diff = abs(SIZE[aa1] - SIZE[aa2]) / 115.0
    
    # Secondary structure propensity differences
    helix_diff = abs(HELIX_PROPENSITY[aa1] - HELIX_PROPENSITY[aa2]) / 1.0
    sheet_diff = abs(SHEET_PROPENSITY[aa1] - SHEET_PROPENSITY[aa2]) / 1.5
    turn_diff = abs(TURN_PROPENSITY[aa1] - TURN_PROPENSITY[aa2]) / 1.0
    
    # Weighted average of all properties
    # Higher weight on charge (conserved in most substitutions)
    # and secondary structure (important for protein folding)
    similarity = 1.0 - (
        0.15 * hydro_diff +
        0.25 * charge_diff +
        0.10 * size_diff +
        0.20 * helix_diff +
        0.20 * sheet_diff +
        0.10 * turn_diff
    )
    
    return max(0.0, min(1.0, similarity))


def get_similar_amino_acids(
    aa: str,
    similarity_threshold: float = 0.7
) -> List[Tuple[str, float]]:
    """
    Get list of amino acids similar to the given amino acid.
    
    Args:
        aa: Target amino acid (single letter code)
        similarity_threshold: Minimum similarity score (0.0 to 1.0)
        
    Returns:
        List of (amino_acid, similarity_score) tuples, sorted by similarity
        
    Reference:
        Uses multi-dimensional biophysical similarity calculation
        inspired by NaNa's molecular biophysics and secondary structure features.
    """
    if aa not in AMINO_ACIDS:
        return []
    
    similar = []
    for other_aa in AMINO_ACIDS:
        if other_aa == aa:
            continue
        
        similarity = calculate_similarity(aa, other_aa)
        if similarity >= similarity_threshold:
            similar.append((other_aa, similarity))
    
    # Sort by similarity (highest first)
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar


def nana_augment(
    sequence: List[str],
    substitution_rate: float,
    similarity_threshold: float = 0.65,
    use_groups: bool = True
) -> List[str]:
    """
    Apply NaNa (Novel Augmentation of New Node Attributes) to a protein sequence.
    
    This function implements the NaNa semantic augmentation technique by
    substituting amino acids with biophysically and structurally similar
    alternatives. The augmentation preserves key properties:
    - Molecular biophysical features (hydrophobicity, charge, size)
    - Secondary structure propensities (α-helix, β-sheet, turn)
    - Overall protein characteristics
    
    Args:
        sequence: List of single-letter amino acid codes
        substitution_rate: Fraction of residues to substitute (0.0 to 1.0)
                          This parameter controls augmentation intensity.
        similarity_threshold: Minimum similarity score for substitutions (0.0 to 1.0)
                            Higher values = more conservative substitutions
        use_groups: If True, use pre-defined biophysical groups for faster lookup
                   If False, use full similarity calculation
                   
    Returns:
        Augmented amino acid sequence (same format as input)
        Returns original sequence if augmentation fails
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
        >>> aug_seq = nana_augment(seq, 0.3)
        >>> # Might produce: ['M', 'D', 'S', 'H', 'L', 'O', 'N', 'V', 'N', 'D']
        >>> # E→D (both negative), T→S (both polar), I→L/V (both hydrophobic)
        
    Reference:
        Paper: "NaNa and MiGu: Semantic Data Augmentation Techniques to 
                Enhance Protein Classification in Graph Neural Networks"
        
        NaNa incorporates:
        - Molecular biophysics features (pKa, functional groups, solvent accessibility)
        - Secondary structure properties (DSSP algorithm: α-helix, β-sheet classes)
        - Node type features (amino acid types, metal ions)
        
        This implementation uses established biochemical property scales
        (Kyte-Doolittle, Chou-Fasman) to achieve semantic augmentation.
        
    Note:
        This implementation is compatible with the augmentation interface used
        in example.py, where augmentation functions take (sequence, intensity)
        and return an augmented sequence.
    """
    if substitution_rate <= 0:
        return sequence
    
    try:
        # Create a copy to avoid modifying original sequence
        augmented_sequence = sequence.copy()
        seq_len = len(augmented_sequence)
        
        if seq_len == 0:
            return sequence
        
        # Calculate number of substitutions
        num_substitutions = max(1, int(substitution_rate * seq_len))
        
        # Randomly select positions to substitute
        positions = random.sample(range(seq_len), min(num_substitutions, seq_len))
        
        # Apply substitutions
        for pos in positions:
            original_aa = augmented_sequence[pos]
            
            # Skip if not a standard amino acid
            if original_aa not in AMINO_ACIDS:
                continue
            
            # Get similar amino acids
            if use_groups and original_aa in BIOPHYSICAL_GROUPS:
                # Fast lookup using pre-defined groups
                candidates = BIOPHYSICAL_GROUPS[original_aa]
                if candidates:
                    augmented_sequence[pos] = random.choice(candidates)
            else:
                # Full similarity calculation
                similar_aas = get_similar_amino_acids(original_aa, similarity_threshold)
                if similar_aas:
                    # Select randomly from similar amino acids
                    # Weight by similarity score
                    weights = [score for _, score in similar_aas]
                    selected = random.choices(
                        [aa for aa, _ in similar_aas],
                        weights=weights,
                        k=1
                    )[0]
                    augmented_sequence[pos] = selected
        
        return augmented_sequence
    
    except Exception:
        # If anything goes wrong, return original sequence
        return sequence
