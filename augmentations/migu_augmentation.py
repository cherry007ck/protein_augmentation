"""
MiGu (Molecular Interactions and Geometric Upgrading) for Protein Sequences

This module implements MiGu augmentation as described in:
"NaNa and MiGu: Semantic Data Augmentation Techniques to Enhance Protein 
Classification in Graph Neural Networks"

MiGu extends NaNa by incorporating molecular interaction information:
- Chemical bonds (hydrogen bonds, disulfide bonds, salt bridges)
- Local sequence context awareness
- Interaction pattern preservation

This implementation adapts the graph-based MiGu technique (with edge attributes)
to sequence-based augmentation using k-mer context analysis.

Reference:
    Paper: "NaNa and MiGu: Semantic Data Augmentation Techniques..."
    arXiv: https://arxiv.org/
"""

import random
from typing import List, Dict, Set, Tuple
from collections import defaultdict


# Standard 20 amino acids
AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


# Critical interaction patterns that should be preserved or respected
# These represent likely molecular interactions in proteins

# Disulfide bond forming residues
DISULFIDE_FORMERS: Set[str] = {'C'}

# Salt bridge forming pairs (oppositely charged residues)
POSITIVE_CHARGED: Set[str] = {'K', 'R', 'H'}
NEGATIVE_CHARGED: Set[str] = {'D', 'E'}

# Hydrogen bond donors
H_BOND_DONORS: Set[str] = {'S', 'T', 'N', 'Q', 'K', 'R', 'H', 'W', 'Y'}

# Hydrogen bond acceptors
H_BOND_ACCEPTORS: Set[str] = {'D', 'E', 'N', 'Q', 'S', 'T', 'Y'}

# Hydrophobic interaction participants
HYDROPHOBIC: Set[str] = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'}

# Aromatic residues (pi-stacking, cation-pi interactions)
AROMATIC: Set[str] = {'F', 'W', 'Y', 'H'}


# Context-aware substitution groups
# These groups consider not just the residue itself, but its likely role
# in molecular interactions based on neighboring residues

# When next to hydrophobic residues, prefer hydrophobic substitutions
HYDROPHOBIC_CONTEXT: Dict[str, List[str]] = {
    'A': ['V', 'I', 'L'],
    'V': ['I', 'L', 'A'],
    'I': ['V', 'L', 'M'],
    'L': ['I', 'V', 'M'],
    'M': ['I', 'L'],
    'F': ['W', 'Y'],
    'W': ['F', 'Y'],
    'Y': ['F', 'W'],
    'P': ['A'],
}

# When next to charged residues, prefer polar/charged substitutions
CHARGED_CONTEXT: Dict[str, List[str]] = {
    'K': ['R', 'H'],
    'R': ['K', 'H'],
    'H': ['K', 'R'],
    'D': ['E', 'N'],
    'E': ['D', 'Q'],
    'N': ['D', 'Q', 'S'],
    'Q': ['E', 'N', 'T'],
    'S': ['T', 'N'],
    'T': ['S', 'Q'],
}

# When next to aromatic residues, prefer aromatic or large hydrophobic
AROMATIC_CONTEXT: Dict[str, List[str]] = {
    'F': ['W', 'Y'],
    'W': ['F', 'Y'],
    'Y': ['F', 'W'],
    'H': ['F', 'Y'],
}


def analyze_local_context(
    sequence: List[str],
    position: int,
    window_size: int = 3
) -> Dict[str, int]:
    """
    Analyze the local sequence context around a position.
    
    Counts occurrences of different amino acid types in a window
    around the target position.
    
    Args:
        sequence: Protein sequence
        position: Target position to analyze
        window_size: Window radius (default: 3, giving 7-residue window)
        
    Returns:
        Dictionary with counts of different residue types in context
    """
    seq_len = len(sequence)
    context = {
        'hydrophobic': 0,
        'charged_positive': 0,
        'charged_negative': 0,
        'polar': 0,
        'aromatic': 0,
        'special': 0,
    }
    
    # Define window boundaries
    start = max(0, position - window_size)
    end = min(seq_len, position + window_size + 1)
    
    # Count residue types in window (excluding target position)
    for i in range(start, end):
        if i == position:
            continue
        
        aa = sequence[i]
        if aa in HYDROPHOBIC:
            context['hydrophobic'] += 1
        if aa in POSITIVE_CHARGED:
            context['charged_positive'] += 1
        if aa in NEGATIVE_CHARGED:
            context['charged_negative'] += 1
        if aa in H_BOND_DONORS or aa in H_BOND_ACCEPTORS:
            context['polar'] += 1
        if aa in AROMATIC:
            context['aromatic'] += 1
        if aa in {'C', 'G', 'P'}:
            context['special'] += 1
    
    return context


def get_context_aware_substitutes(
    original_aa: str,
    context: Dict[str, int]
) -> List[str]:
    """
    Get amino acid substitutes based on local sequence context.
    
    Args:
        original_aa: Original amino acid to substitute
        context: Local context analysis from analyze_local_context()
        
    Returns:
        List of suitable amino acid substitutes considering context
    """
    if original_aa not in AMINO_ACIDS:
        return []
    
    # Determine dominant context type
    dominant_context = max(context.items(), key=lambda x: x[1])[0]
    
    # Select substitution group based on context
    candidates = []
    
    if dominant_context == 'hydrophobic':
        if original_aa in HYDROPHOBIC_CONTEXT:
            candidates = HYDROPHOBIC_CONTEXT[original_aa]
    
    elif dominant_context in ['charged_positive', 'charged_negative', 'polar']:
        if original_aa in CHARGED_CONTEXT:
            candidates = CHARGED_CONTEXT[original_aa]
    
    elif dominant_context == 'aromatic':
        if original_aa in AROMATIC_CONTEXT:
            candidates = AROMATIC_CONTEXT[original_aa]
    
    # Fallback to basic biophysical groups if no context-specific match
    if not candidates:
        # Use NaNa-style groups as fallback
        biophysical_fallback = {
            'A': ['V'], 'V': ['A', 'I'], 'I': ['V', 'L'], 'L': ['I'],
            'M': ['L'], 'F': ['Y'], 'W': ['Y'], 'Y': ['F'],
            'S': ['T'], 'T': ['S'], 'N': ['Q'], 'Q': ['N'],
            'K': ['R'], 'R': ['K'], 'H': ['K'], 'D': ['E'], 'E': ['D'],
            'C': ['S'], 'G': ['A'], 'P': ['A'],
        }
        candidates = biophysical_fallback.get(original_aa, [])
    
    return candidates


def preserve_critical_interactions(
    sequence: List[str],
    position: int,
    candidate_aa: str
) -> bool:
    """
    Check if substitution preserves critical interaction patterns.
    
    Args:
        sequence: Protein sequence
        position: Position being substituted
        candidate_aa: Candidate amino acid for substitution
        
    Returns:
        True if substitution is safe (preserves critical patterns)
    """
    original_aa = sequence[position]
    seq_len = len(sequence)
    
    # Rule 1: Preserve cysteine pairs (disulfide bonds)
    # If original is C and there's another C nearby, keep it as C
    if original_aa == 'C':
        # Check for nearby cysteines (within 10 residues)
        window_start = max(0, position - 10)
        window_end = min(seq_len, position + 11)
        
        for i in range(window_start, window_end):
            if i != position and sequence[i] == 'C':
                # There's another cysteine nearby - only allow C→C
                return candidate_aa == 'C'
    
    # Rule 2: Preserve salt bridge potential
    # If there's a charged residue nearby, maintain charge complementarity
    if original_aa in POSITIVE_CHARGED or original_aa in NEGATIVE_CHARGED:
        # Check immediate neighbors (±3 residues)
        for offset in [-3, -2, -1, 1, 2, 3]:
            neighbor_pos = position + offset
            if 0 <= neighbor_pos < seq_len:
                neighbor = sequence[neighbor_pos]
                
                # If positive next to negative, preserve the interaction potential
                if (original_aa in POSITIVE_CHARGED and neighbor in NEGATIVE_CHARGED) or \
                   (original_aa in NEGATIVE_CHARGED and neighbor in POSITIVE_CHARGED):
                    # Candidate should maintain charge of same sign
                    if original_aa in POSITIVE_CHARGED:
                        return candidate_aa in POSITIVE_CHARGED
                    else:
                        return candidate_aa in NEGATIVE_CHARGED
    
    # Rule 3: Preserve aromatic clusters (π-stacking)
    # If surrounded by aromatic residues, maintain aromatic character
    if original_aa in AROMATIC:
        aromatic_neighbors = 0
        for offset in [-1, 1]:
            neighbor_pos = position + offset
            if 0 <= neighbor_pos < seq_len:
                if sequence[neighbor_pos] in AROMATIC:
                    aromatic_neighbors += 1
        
        # If both neighbors are aromatic, keep it aromatic
        if aromatic_neighbors >= 2:
            return candidate_aa in AROMATIC
    
    # If no critical pattern detected, substitution is safe
    return True


def migu_augment(
    sequence: List[str],
    substitution_rate: float,
    context_window: int = 3,
    preserve_interactions: bool = True
) -> List[str]:
    """
    Apply MiGu (Molecular Interactions and Geometric Upgrading) to a protein sequence.
    
    MiGu extends NaNa by incorporating molecular interaction information.
    This implementation uses local sequence context to make substitutions that
    respect likely molecular interactions:
    
    - Analyzes k-mer context around each position
    - Preserves critical interaction patterns (disulfide bonds, salt bridges)
    - Makes context-aware substitutions based on neighboring residues
    
    Args:
        sequence: List of single-letter amino acid codes
        substitution_rate: Fraction of residues to substitute (0.0 to 1.0)
                          This parameter controls augmentation intensity.
        context_window: Window radius for context analysis (default: 3)
                       Larger windows consider more distant interactions
        preserve_interactions: If True, preserve critical interaction patterns
                              (disulfide bonds, salt bridges, aromatic clusters)
                              
    Returns:
        Augmented amino acid sequence (same format as input)
        Returns original sequence if augmentation fails
        
    Example:
        >>> seq = ['C', 'Y', 'S', 'K', 'D', 'L', 'F', 'C']
        >>> aug_seq = migu_augment(seq, 0.3)
        >>> # Preserves: C-C pair (disulfide), K-D pair (salt bridge)
        >>> # May change: Y→F (aromatic context), S→T (polar), L→I (hydrophobic)
        
    Reference:
        Paper: "NaNa and MiGu: Semantic Data Augmentation Techniques to 
                Enhance Protein Classification in Graph Neural Networks"
        
        MiGu builds upon NaNa by incorporating edge attributes:
        - Chemical bonds (hydrogen bonds, disulfide bonds via Baker-Hubbard Theory)
        - Empirical Binary Function for bond-type characterization
        - Molecular interaction information for semantic augmentation
        
        This sequence-based implementation uses context analysis and
        interaction pattern preservation to capture MiGu's principles.
        
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
        
        # Apply context-aware substitutions
        for pos in positions:
            original_aa = augmented_sequence[pos]
            
            # Skip if not a standard amino acid
            if original_aa not in AMINO_ACIDS:
                continue
            
            # Analyze local context
            context = analyze_local_context(augmented_sequence, pos, context_window)
            
            # Get context-aware substitution candidates
            candidates = get_context_aware_substitutes(original_aa, context)
            
            if not candidates:
                # No suitable candidates found, skip this position
                continue
            
            # Filter candidates based on interaction preservation
            if preserve_interactions:
                valid_candidates = [
                    aa for aa in candidates 
                    if preserve_critical_interactions(augmented_sequence, pos, aa)
                ]
                
                # If all candidates filtered out, keep original
                if not valid_candidates:
                    continue
                
                candidates = valid_candidates
            
            # Select a random candidate
            augmented_sequence[pos] = random.choice(candidates)
        
        return augmented_sequence
    
    except Exception:
        # If anything goes wrong, return original sequence
        return sequence
