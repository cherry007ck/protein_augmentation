"""
Simplified Retrieved Sequence Augmentation (RSA) for Protein Sequences

This module implements a lightweight version of Retrieved Sequence Augmentation
inspired by the paper:
"Retrieved Sequence Augmentation for Protein Representation Learning"
by Chang et al. (https://github.com/chang-github-00/RSA)

The full RSA implementation uses FAISS indexing to retrieve similar sequences
from large protein databases (Pfam, UniRef). This simplified version simulates
the retrieval process by generating "pseudo-homologous" sequences through
conservative amino acid substitutions, maintaining the spirit of RSA without
requiring external databases.

Key differences from full RSA:
- No external sequence database required
- No FAISS indexing needed
- Uses conservative mutations instead of actual sequence retrieval
- Maintains compatibility with lightweight augmentation framework

Reference:
    Chang, Y. et al. (2023)
    "Retrieved Sequence Augmentation for Protein Representation Learning"
    arXiv preprint
    https://arxiv.org/abs/2302.12563
"""

import random
from typing import List, Dict


# Amino acid groups based on biochemical properties
# Used for conservative substitutions that preserve protein function
AMINO_ACID_GROUPS: Dict[str, List[str]] = {
    # Hydrophobic amino acids
    'A': ['V', 'I', 'L'],  # Alanine -> small hydrophobic
    'V': ['A', 'I', 'L'],  # Valine -> branched hydrophobic
    'I': ['V', 'L', 'M'],  # Isoleucine -> large hydrophobic
    'L': ['V', 'I', 'M'],  # Leucine -> large hydrophobic
    'M': ['L', 'I', 'F'],  # Methionine -> sulfur-containing hydrophobic
    'F': ['W', 'Y', 'M'],  # Phenylalanine -> aromatic hydrophobic
    'W': ['F', 'Y'],       # Tryptophan -> large aromatic
    'P': ['A', 'G'],       # Proline -> helix breaker (special case)
    
    # Polar uncharged amino acids
    'S': ['T', 'C', 'N'],  # Serine -> small polar
    'T': ['S', 'C', 'N'],  # Threonine -> hydroxyl-containing
    'C': ['S', 'T'],       # Cysteine -> sulfur-containing (conservative: fewer options)
    'N': ['Q', 'S', 'T'],  # Asparagine -> amide-containing
    'Q': ['N', 'E', 'K'],  # Glutamine -> amide-containing
    'Y': ['F', 'W', 'H'],  # Tyrosine -> aromatic polar
    
    # Positively charged amino acids
    'K': ['R', 'Q', 'H'],  # Lysine -> positive charge
    'R': ['K', 'H'],       # Arginine -> positive charge
    'H': ['K', 'R', 'Y'],  # Histidine -> aromatic positive (pH-dependent)
    
    # Negatively charged amino acids
    'D': ['E', 'N'],       # Aspartic acid -> negative charge
    'E': ['D', 'Q'],       # Glutamic acid -> negative charge
    
    # Special amino acids
    'G': ['A', 'S'],       # Glycine -> smallest, most flexible
}


def generate_conservative_mutation(amino_acid: str) -> str:
    """
    Generate a conservative mutation for a given amino acid.
    
    Conservative mutations replace an amino acid with one that has
    similar biochemical properties (size, charge, hydrophobicity).
    This simulates natural evolution where such mutations are
    more likely to preserve protein function.
    
    Args:
        amino_acid: Single-letter amino acid code
        
    Returns:
        A biochemically similar amino acid, or the original if
        no suitable substitution exists
        
    Example:
        >>> generate_conservative_mutation('L')
        'V'  # or 'I', 'M' - all large hydrophobic
        
    Reference:
        Conservative substitutions are based on the BLOSUM62 matrix
        and standard biochemical classifications.
    """
    if amino_acid in AMINO_ACID_GROUPS:
        # Randomly choose from conservative substitutions
        substitutes = AMINO_ACID_GROUPS[amino_acid]
        if substitutes:
            return random.choice(substitutes)
    
    # If no conservative substitution available, return original
    return amino_acid


def rsa_augment(sequence: List[str], intensity: float) -> List[str]:
    """
    Apply simplified Retrieved Sequence Augmentation (RSA) to a protein sequence.
    
    This function simulates the RSA retrieval process by generating a
    "pseudo-homologous" sequence through conservative amino acid substitutions.
    The generated sequence mimics what would be retrieved from a database
    of evolutionarily related proteins.
    
    The intensity parameter controls how divergent the "retrieved" sequence is:
    - Low intensity (e.g., 0.1): Very similar to query (recent homolog)
    - High intensity (e.g., 0.5): More divergent (distant homolog)
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Fraction of residues to mutate (0.0 to 1.0)
                  Controls the evolutionary distance of the "retrieved" sequence
                  
    Returns:
        A conservatively mutated sequence (same format as input)
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
        >>> aug_seq = rsa_augment(seq, 0.3)
        >>> # Approximately 30% of residues will be conservatively mutated
        >>> # L->I (both hydrophobic), E->D (both negative), etc.
        
    Reference:
        This implementation is inspired by:
        Chang, Y. et al. (2023) "Retrieved Sequence Augmentation for 
        Protein Representation Learning"
        
        The full RSA uses FAISS to retrieve actual homologous sequences.
        This simplified version generates synthetic homologs using the
        same principle: sequence variation while preserving function.
        
    Note:
        This implementation is compatible with the augmentation interface used
        in example.py, where augmentation functions take (sequence, intensity)
        and return an augmented sequence.
    """
    if intensity <= 0:
        return sequence
    
    # Ensure intensity doesn't exceed 1.0
    intensity = min(intensity, 1.0)
    
    # Create a copy to avoid modifying the original
    augmented_sequence = sequence.copy()
    seq_len = len(augmented_sequence)
    
    if seq_len == 0:
        return sequence
    
    # Calculate number of positions to mutate
    num_mutations = max(1, int(intensity * seq_len))
    
    # Randomly select positions to mutate
    positions_to_mutate = random.sample(range(seq_len), min(num_mutations, seq_len))
    
    # Apply conservative mutations
    for pos in positions_to_mutate:
        original_aa = augmented_sequence[pos]
        mutated_aa = generate_conservative_mutation(original_aa)
        augmented_sequence[pos] = mutated_aa
    
    return augmented_sequence


def rsa_augment_with_original(sequence: List[str], intensity: float) -> List[str]:
    """
    Alternative RSA augmentation that can return the original sequence.
    
    With some probability, returns the exact original sequence (simulating
    retrieval of an identical sequence). Otherwise, applies conservative
    mutations. This adds more diversity to the augmentation strategy.
    
    Args:
        sequence: List of single-letter amino acid codes
        intensity: Fraction of residues to mutate when mutation is applied
                  
    Returns:
        Either the original sequence or a conservatively mutated version
        
    Example:
        >>> seq = ['A', 'R', 'G', 'L']
        >>> aug_seq = rsa_augment_with_original(seq, 0.3)
        >>> # 50% chance: returns original ['A', 'R', 'G', 'L']
        >>> # 50% chance: returns mutated sequence
        
    Reference:
        In real RSA, the k retrieved sequences include both very similar
        and somewhat divergent homologs. This function simulates that
        diversity by sometimes returning the exact query sequence.
    """
    # 50% chance to return original sequence (like retrieving identical match)
    if random.random() < 0.5:
        return sequence
    else:
        return rsa_augment(sequence, intensity)
