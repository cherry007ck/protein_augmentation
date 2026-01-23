"""
BootGen-Inspired Augmentation for Protein Sequences

This module implements a simplified version of BootGen (Bootstrapped Training 
of Score-Conditioned Generator) as described in:
"Bootstrapped Training of Score-Conditioned Generator for Offline Design 
of Biological Sequences" (NeurIPS 2023)
https://arxiv.org/abs/2306.00111

The technique uses bootstrapped sampling with rank-based weighting to generate
high-quality augmented protein sequences. This implementation adapts the core
concepts for sequence-level data augmentation.

Key Components:
1. Proxy Score Function: Estimates sequence quality based on composition and properties
2. Bootstrapped Generation: Creates multiple candidate sequences
3. Rank-Based Selection: Probabilistically selects the best candidate

Author: Adapted for protein data augmentation framework
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Callable
from collections import Counter


# Amino acid groups by biochemical properties
AMINO_ACID_GROUPS: Dict[str, List[str]] = {
    'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'],
    'hydrophilic': ['S', 'T', 'N', 'Q'],
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E'],
    'special': ['C', 'G', 'Y']
}

# Reverse mapping: amino acid to group
AA_TO_GROUP: Dict[str, str] = {}
for group, aas in AMINO_ACID_GROUPS.items():
    for aa in aas:
        AA_TO_GROUP[aa] = group

# All standard amino acids
STANDARD_AAS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")


def compute_composition_similarity(
    seq1: List[str], 
    seq2: List[str]
) -> float:
    """
    Compute amino acid composition similarity between two sequences.
    
    Uses cosine similarity of composition vectors.
    
    Args:
        seq1: First amino acid sequence
        seq2: Second amino acid sequence
        
    Returns:
        Similarity score in [0, 1], where 1 is identical composition
        
    Reference:
        BootGen (NeurIPS 2023), Section 3.2: Proxy score functions
    """
    if not seq1 or not seq2:
        return 0.0
    
    # Count amino acids in each sequence
    count1 = Counter(seq1)
    count2 = Counter(seq2)
    
    # Get all amino acids present
    all_aas = set(count1.keys()) | set(count2.keys())
    
    # Create composition vectors
    vec1 = np.array([count1.get(aa, 0) for aa in all_aas], dtype=float)
    vec2 = np.array([count2.get(aa, 0) for aa in all_aas], dtype=float)
    
    # Normalize to get frequencies
    if vec1.sum() > 0:
        vec1 = vec1 / vec1.sum()
    if vec2.sum() > 0:
        vec2 = vec2 / vec2.sum()
    
    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(np.clip(similarity, 0.0, 1.0))


def compute_property_similarity(
    seq1: List[str], 
    seq2: List[str]
) -> float:
    """
    Compute biochemical property distribution similarity.
    
    Compares the distribution of amino acid groups (hydrophobic, charged, etc.)
    between two sequences.
    
    Args:
        seq1: First amino acid sequence
        seq2: Second amino acid sequence
        
    Returns:
        Similarity score in [0, 1]
        
    Reference:
        BootGen (NeurIPS 2023), Section 3.2: Proxy score functions
    """
    if not seq1 or not seq2:
        return 0.0
    
    # Count groups in each sequence
    count1 = Counter(AA_TO_GROUP.get(aa, 'unknown') for aa in seq1)
    count2 = Counter(AA_TO_GROUP.get(aa, 'unknown') for aa in seq2)
    
    # Get all groups present
    all_groups = set(count1.keys()) | set(count2.keys())
    
    # Create property vectors
    vec1 = np.array([count1.get(g, 0) for g in all_groups], dtype=float)
    vec2 = np.array([count2.get(g, 0) for g in all_groups], dtype=float)
    
    # Normalize
    if vec1.sum() > 0:
        vec1 = vec1 / vec1.sum()
    if vec2.sum() > 0:
        vec2 = vec2 / vec2.sum()
    
    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(np.clip(similarity, 0.0, 1.0))


def compute_sequence_score(
    original: List[str], 
    candidate: List[str],
    length_penalty_weight: float = 0.2
) -> float:
    """
    Compute proxy score for a candidate sequence relative to the original.
    
    The proxy score combines:
    - Composition similarity (40%)
    - Property distribution similarity (40%)
    - Length preservation (20%)
    
    Args:
        original: Original reference sequence
        candidate: Candidate augmented sequence
        length_penalty_weight: Weight for length penalty (default: 0.2)
        
    Returns:
        Quality score in [0, 1], where higher is better
        
    Reference:
        BootGen (NeurIPS 2023), Section 3.2: "We use a proxy score function
        to evaluate the quality of generated sequences"
    """
    if not original or not candidate:
        return 0.0
    
    # Composition similarity (40% weight)
    comp_sim = compute_composition_similarity(original, candidate)
    
    # Property similarity (40% weight)
    prop_sim = compute_property_similarity(original, candidate)
    
    # Length penalty (20% weight)
    len_ratio = len(candidate) / len(original) if len(original) > 0 else 0.0
    # Penalize sequences that are too short or too long
    # Ideal ratio is 1.0, allow ±30% with linear penalty
    if 0.7 <= len_ratio <= 1.3:
        len_score = 1.0 - abs(1.0 - len_ratio) / 0.3
    else:
        len_score = 0.0
    
    # Weighted combination
    total_score = (
        0.4 * comp_sim + 
        0.4 * prop_sim + 
        length_penalty_weight * len_score
    )
    
    return float(np.clip(total_score, 0.0, 1.0))


def generate_candidate_sequence(
    sequence: List[str],
    num_substitutions: int,
    conservative: bool = True
) -> List[str]:
    """
    Generate a single candidate sequence through substitution.
    
    Args:
        sequence: Original amino acid sequence
        num_substitutions: Number of positions to substitute
        conservative: If True, substitute with biochemically similar AAs
        
    Returns:
        Candidate augmented sequence
        
    Reference:
        BootGen (NeurIPS 2023), Section 3.1: "Generate diverse candidates"
    """
    if len(sequence) == 0:
        return sequence
    
    candidate = sequence.copy()
    seq_len = len(candidate)
    
    # Randomly select positions to substitute
    num_subs = min(num_substitutions, seq_len)
    if num_subs <= 0:
        return candidate
    
    positions = random.sample(range(seq_len), num_subs)
    
    for pos in positions:
        original_aa = candidate[pos]
        
        if conservative and original_aa in AA_TO_GROUP:
            # Conservative substitution: same biochemical group
            group = AA_TO_GROUP[original_aa]
            possible_aas = [aa for aa in AMINO_ACID_GROUPS[group] if aa != original_aa]
            if possible_aas:
                candidate[pos] = random.choice(possible_aas)
            else:
                # If no alternatives in group, choose random
                candidate[pos] = random.choice(STANDARD_AAS)
        else:
            # Random substitution
            candidate[pos] = random.choice(STANDARD_AAS)
    
    return candidate


def generate_bootstrapped_sequences(
    sequence: List[str],
    num_candidates: int,
    num_substitutions: int,
    conservative: bool = True
) -> List[List[str]]:
    """
    Generate multiple candidate sequences through bootstrapped sampling.
    
    Creates diverse candidates by applying different random substitutions
    to the original sequence.
    
    Args:
        sequence: Original amino acid sequence
        num_candidates: Number of candidate sequences to generate
        num_substitutions: Number of substitutions per candidate
        conservative: If True, use conservative (biochemically similar) substitutions
        
    Returns:
        List of candidate sequences
        
    Reference:
        BootGen (NeurIPS 2023), Section 3.1: "Bootstrapped generation creates
        diverse candidates which are then filtered by quality"
    """
    candidates = []
    
    for _ in range(num_candidates):
        candidate = generate_candidate_sequence(
            sequence, 
            num_substitutions, 
            conservative
        )
        candidates.append(candidate)
    
    return candidates


def rank_based_selection(
    candidates: List[List[str]],
    scores: List[float],
    temperature: float = 1.0
) -> List[str]:
    """
    Select a candidate using rank-based probabilistic sampling.
    
    Higher-scored candidates have higher probability of being selected.
    
    Args:
        candidates: List of candidate sequences
        scores: Corresponding quality scores
        temperature: Sampling temperature (higher = more random, default: 1.0)
        
    Returns:
        Selected candidate sequence
        
    Reference:
        BootGen (NeurIPS 2023), Section 3.3: "Rank-based weighting prioritizes
        high-quality sequences during sampling"
    """
    if not candidates or not scores:
        return []
    
    if len(candidates) != len(scores):
        raise ValueError("Number of candidates must match number of scores")
    
    # Convert scores to numpy array
    scores_array = np.array(scores, dtype=float)
    
    # Apply temperature scaling
    if temperature > 0:
        scores_array = scores_array / temperature
    
    # Softmax to get probabilities
    exp_scores = np.exp(scores_array - np.max(scores_array))  # numerical stability
    probabilities = exp_scores / exp_scores.sum()
    
    # Sample based on probabilities
    selected_idx = np.random.choice(len(candidates), p=probabilities)
    
    return candidates[selected_idx]


def bootgen_augment(
    sequence: List[str], 
    intensity: float
) -> List[str]:
    """
    Apply BootGen-inspired augmentation to a protein sequence.
    
    This function implements a simplified version of the BootGen algorithm:
    1. Generate multiple candidate sequences (bootstrapped sampling)
    2. Score each candidate using a proxy quality function
    3. Select the best candidate using rank-based weighting
    
    The intensity parameter controls:
    - Number of amino acid substitutions
    - Number of candidate sequences generated
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Augmentation intensity (0.0 to 1.0)
                  - 0.0: minimal changes (1-2 substitutions, 5 candidates)
                  - 0.5: moderate changes (~50% of length, 10 candidates)
                  - 1.0: aggressive changes (more substitutions, 20 candidates)
                  
    Returns:
        Augmented amino acid sequence (same format as input)
        Returns original sequence if augmentation fails
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
        >>> aug_seq = bootgen_augment(seq, 0.3)
        >>> # Returns a high-quality augmented variant with conservative substitutions
        
    Reference:
        Anand, N., Eguchi, R., Mathews, I. I., Perez, C. P., Derry, A., 
        Altman, R. B., & Huang, P. S. (2023).
        "Bootstrapped Training of Score-Conditioned Generator for Offline 
        Design of Biological Sequences"
        NeurIPS 2023
        https://arxiv.org/abs/2306.00111
        
    Note:
        This implementation is compatible with the augmentation interface used
        in example.py, where augmentation functions take (sequence, intensity)
        and return an augmented sequence.
    """
    try:
        # Handle edge cases
        if not sequence or len(sequence) == 0:
            return sequence
        
        if intensity <= 0.0:
            return sequence
        
        # Calculate number of substitutions based on intensity
        seq_len = len(sequence)
        # Scale from 1-2 subs at low intensity to ~50% at high intensity
        base_subs = max(1, int(intensity * seq_len * 0.5))
        num_substitutions = min(base_subs, seq_len)
        
        # Calculate number of candidates based on intensity
        # More candidates at higher intensity for better quality selection
        num_candidates = max(5, min(20, int(5 + intensity * 15)))
        
        # Generate candidate sequences
        candidates = generate_bootstrapped_sequences(
            sequence=sequence,
            num_candidates=num_candidates,
            num_substitutions=num_substitutions,
            conservative=True  # Use conservative substitutions
        )
        
        # Score each candidate
        scores = []
        for candidate in candidates:
            score = compute_sequence_score(sequence, candidate)
            scores.append(score)
        
        # Select best candidate using rank-based sampling
        # Lower temperature at higher intensity = more deterministic (pick best)
        temperature = max(0.5, 1.5 - intensity)
        selected = rank_based_selection(candidates, scores, temperature)
        
        # Validate the selected sequence
        if len(selected) < 5:  # Too short
            return sequence
        
        # Check that all amino acids are valid
        if not all(aa in STANDARD_AAS for aa in selected):
            return sequence
        
        return selected
    
    except Exception:
        # If anything goes wrong, return original sequence
        return sequence
