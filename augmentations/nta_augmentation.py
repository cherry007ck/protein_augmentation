"""
Nucleotide Augmentation (NTA) for Protein Sequences

This module implements Nucleotide Augmentation as described in:
"Nucleotide augmentation for machine learning-guided protein engineering"
by Minot & Reddy, 2022 (https://doi.org/10.1093/bioadv/vbac094)

The technique performs back-translation from amino acids to DNA codons,
applies synonymous codon substitutions, then translates back to amino acids.
This creates sequence diversity while preserving the amino acid sequence identity.
"""

import random
from typing import List, Dict


# Standard genetic code: DNA codon to amino acid mapping
CODON_TO_AMINO_ACID: Dict[str, str] = {
    # Alanine (A)
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Arginine (R)
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Asparagine (N)
    'AAT': 'N', 'AAC': 'N',
    # Aspartic acid (D)
    'GAT': 'D', 'GAC': 'D',
    # Cysteine (C)
    'TGT': 'C', 'TGC': 'C',
    # Glutamine (Q)
    'CAA': 'Q', 'CAG': 'Q',
    # Glutamic acid (E)
    'GAA': 'E', 'GAG': 'E',
    # Glycine (G)
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    # Histidine (H)
    'CAT': 'H', 'CAC': 'H',
    # Isoleucine (I)
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    # Leucine (L)
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'TTA': 'L', 'TTG': 'L',
    # Lysine (K)
    'AAA': 'K', 'AAG': 'K',
    # Methionine (M)
    'ATG': 'M',
    # Phenylalanine (F)
    'TTT': 'F', 'TTC': 'F',
    # Proline (P)
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Serine (S)
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    # Threonine (T)
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Tryptophan (W)
    'TGG': 'W',
    # Tyrosine (Y)
    'TAT': 'Y', 'TAC': 'Y',
    # Valine (V)
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
}

# Reverse mapping: amino acid to list of synonymous codons
AMINO_ACID_TO_CODONS: Dict[str, List[str]] = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'],
    'C': ['TGT', 'TGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'L': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
}

# Synonymous codon substitution dictionary: maps each codon to its synonyms
SYNONYMOUS_CODONS: Dict[str, List[str]] = {
    # Alanine (A)
    'GCT': ['GCC', 'GCA', 'GCG'],
    'GCC': ['GCT', 'GCA', 'GCG'],
    'GCA': ['GCT', 'GCC', 'GCG'],
    'GCG': ['GCT', 'GCC', 'GCA'],
    # Arginine (R)
    'CGT': ['CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'CGC': ['CGT', 'CGA', 'CGG', 'AGA', 'AGG'],
    'CGA': ['CGT', 'CGC', 'CGG', 'AGA', 'AGG'],
    'CGG': ['CGT', 'CGC', 'CGA', 'AGA', 'AGG'],
    'AGA': ['CGT', 'CGC', 'CGA', 'CGG', 'AGG'],
    'AGG': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA'],
    # Asparagine (N)
    'AAT': ['AAC'],
    'AAC': ['AAT'],
    # Aspartic acid (D)
    'GAT': ['GAC'],
    'GAC': ['GAT'],
    # Cysteine (C)
    'TGT': ['TGC'],
    'TGC': ['TGT'],
    # Glutamine (Q)
    'CAA': ['CAG'],
    'CAG': ['CAA'],
    # Glutamic acid (E)
    'GAA': ['GAG'],
    'GAG': ['GAA'],
    # Glycine (G)
    'GGT': ['GGC', 'GGA', 'GGG'],
    'GGC': ['GGT', 'GGA', 'GGG'],
    'GGA': ['GGT', 'GGC', 'GGG'],
    'GGG': ['GGT', 'GGC', 'GGA'],
    # Histidine (H)
    'CAT': ['CAC'],
    'CAC': ['CAT'],
    # Isoleucine (I)
    'ATT': ['ATC', 'ATA'],
    'ATC': ['ATT', 'ATA'],
    'ATA': ['ATT', 'ATC'],
    # Leucine (L)
    'CTT': ['CTC', 'CTA', 'CTG', 'TTA', 'TTG'],
    'CTC': ['CTT', 'CTA', 'CTG', 'TTA', 'TTG'],
    'CTA': ['CTT', 'CTC', 'CTG', 'TTA', 'TTG'],
    'CTG': ['CTT', 'CTC', 'CTA', 'TTA', 'TTG'],
    'TTA': ['CTT', 'CTC', 'CTA', 'CTG', 'TTG'],
    'TTG': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA'],
    # Lysine (K)
    'AAA': ['AAG'],
    'AAG': ['AAA'],
    # Methionine (M) - only one codon
    'ATG': ['ATG'],
    # Phenylalanine (F)
    'TTT': ['TTC'],
    'TTC': ['TTT'],
    # Proline (P)
    'CCT': ['CCC', 'CCA', 'CCG'],
    'CCC': ['CCT', 'CCA', 'CCG'],
    'CCA': ['CCT', 'CCC', 'CCG'],
    'CCG': ['CCT', 'CCC', 'CCA'],
    # Serine (S)
    'TCT': ['TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'TCC': ['TCT', 'TCA', 'TCG', 'AGT', 'AGC'],
    'TCA': ['TCT', 'TCC', 'TCG', 'AGT', 'AGC'],
    'TCG': ['TCT', 'TCC', 'TCA', 'AGT', 'AGC'],
    'AGT': ['TCT', 'TCC', 'TCA', 'TCG', 'AGC'],
    'AGC': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT'],
    # Threonine (T)
    'ACT': ['ACC', 'ACA', 'ACG'],
    'ACC': ['ACT', 'ACA', 'ACG'],
    'ACA': ['ACT', 'ACC', 'ACG'],
    'ACG': ['ACT', 'ACC', 'ACA'],
    # Tryptophan (W) - only one codon
    'TGG': ['TGG'],
    # Tyrosine (Y)
    'TAT': ['TAC'],
    'TAC': ['TAT'],
    # Valine (V)
    'GTT': ['GTC', 'GTA', 'GTG'],
    'GTC': ['GTT', 'GTA', 'GTG'],
    'GTA': ['GTT', 'GTC', 'GTG'],
    'GTG': ['GTT', 'GTC', 'GTA'],
}


def back_translate(amino_acid_sequence: List[str]) -> List[str]:
    """
    Back-translate an amino acid sequence to DNA codons.
    
    Randomly selects one of the synonymous codons for each amino acid.
    
    Args:
        amino_acid_sequence: List of single-letter amino acid codes
        
    Returns:
        List of DNA codons (3-letter strings)
        
    Reference:
        Minot & Reddy 2022, Section 2.2: "Nucleotide Augmentation"
    """
    codon_sequence = []
    for aa in amino_acid_sequence:
        if aa in AMINO_ACID_TO_CODONS:
            # Randomly select one synonymous codon
            codon = random.choice(AMINO_ACID_TO_CODONS[aa])
            codon_sequence.append(codon)
        else:
            # Handle unknown amino acids by returning empty sequence
            # This will cause the augmentation to fail gracefully
            return []
    return codon_sequence


def forward_translate(codon_sequence: List[str]) -> List[str]:
    """
    Translate DNA codons to amino acids.
    
    Args:
        codon_sequence: List of DNA codons (3-letter strings)
        
    Returns:
        List of single-letter amino acid codes
        
    Reference:
        Minot & Reddy 2022, Section 2.2: "Nucleotide Augmentation"
    """
    amino_acid_sequence = []
    for codon in codon_sequence:
        if codon in CODON_TO_AMINO_ACID:
            aa = CODON_TO_AMINO_ACID[codon]
            amino_acid_sequence.append(aa)
        else:
            # Unknown codon - return empty to fail gracefully
            return []
    return amino_acid_sequence


def apply_synonymous_substitutions(
    codon_sequence: List[str],
    substitution_rate: float
) -> List[str]:
    """
    Apply synonymous codon substitutions to a DNA sequence.
    
    Randomly substitutes codons with their synonymous alternatives
    at the specified rate.
    
    Args:
        codon_sequence: List of DNA codons
        substitution_rate: Fraction of codons to substitute (0.0 to 1.0)
        
    Returns:
        List of DNA codons with synonymous substitutions applied
        
    Reference:
        Minot & Reddy 2022, Section 2.2: "Nucleotide Augmentation"
        The parameter 'substitution_rate' corresponds to the 'subst_frac'
        parameter in the original implementation.
    """
    if substitution_rate <= 0:
        return codon_sequence
    
    augmented_sequence = codon_sequence.copy()
    num_codons = len(augmented_sequence)
    
    # Determine number of substitutions
    num_substitutions = max(1, int(substitution_rate * num_codons))
    
    # Randomly select positions to substitute
    positions = random.sample(range(num_codons), min(num_substitutions, num_codons))
    
    # Apply synonymous substitutions
    for pos in positions:
        original_codon = augmented_sequence[pos]
        if original_codon in SYNONYMOUS_CODONS:
            synonyms = SYNONYMOUS_CODONS[original_codon]
            if synonyms and synonyms != [original_codon]:  # Has synonyms other than itself
                # Replace with a random synonymous codon
                augmented_sequence[pos] = random.choice(synonyms)
    
    return augmented_sequence


def nucleotide_augment(sequence: List[str], substitution_rate: float) -> List[str]:
    """
    Apply Nucleotide Augmentation (NTA) to a protein sequence.
    
    This function implements the NTA technique from Minot & Reddy 2022:
    1. Back-translate amino acids to DNA codons
    2. Apply synonymous codon substitutions
    3. Forward-translate back to amino acids
    
    The result is a protein sequence that is identical at the amino acid level
    (assuming no errors), but has variation at the nucleotide level. This technique
    is particularly useful for protein engineering tasks where sequence diversity
    is beneficial but functional constraints must be preserved.
    
    Args:
        sequence: List of single-letter amino acid codes
        substitution_rate: Fraction of codons to substitute (0.0 to 1.0)
                          This parameter controls the intensity of augmentation.
                          Higher values create more diversity.
                          
    Returns:
        Augmented amino acid sequence (same format as input)
        Returns original sequence if augmentation fails
        
    Example:
        >>> seq = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
        >>> aug_seq = nucleotide_augment(seq, 0.3)
        >>> # aug_seq should be the same amino acids, but derived from
        >>> # different synonymous codons at the nucleotide level
        
    Reference:
        Minot, M. and Reddy, S.T. (2022)
        "Nucleotide augmentation for machine learning-guided protein engineering"
        Bioinformatics Advances, vbac094
        https://doi.org/10.1093/bioadv/vbac094
        
    Note:
        This implementation is compatible with the augmentation interface used
        in example.py, where augmentation functions take (sequence, intensity)
        and return an augmented sequence.
    """
    try:
        # Step 1: Back-translate to DNA codons
        codon_sequence = back_translate(sequence)
        if not codon_sequence:
            return sequence  # Failed to back-translate
        
        # Step 2: Apply synonymous substitutions
        augmented_codons = apply_synonymous_substitutions(codon_sequence, substitution_rate)
        
        # Step 3: Forward-translate back to amino acids
        augmented_sequence = forward_translate(augmented_codons)
        if not augmented_sequence:
            return sequence  # Failed to translate
        
        # Sanity check: ensure length is preserved
        if len(augmented_sequence) != len(sequence):
            return sequence
        
        return augmented_sequence
    
    except Exception:
        # If anything goes wrong, return original sequence
        return sequence
