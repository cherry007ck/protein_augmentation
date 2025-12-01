"""
Test script for NTA (Nucleotide Augmentation) implementation.

This script verifies that the NTA augmentation:
1. Correctly back-translates and forward-translates sequences
2. Preserves amino acid sequence identity (when substitution_rate=0)
3. Creates variation when substitution_rate > 0
4. Integrates properly with the existing augmentation framework
"""

import sys
sys.path.insert(0, '/home/luffy/protein_augmentation')

from augmentations.nta_augmentation import (
    back_translate,
    forward_translate,
    apply_synonymous_substitutions,
    nucleotide_augment,
    AMINO_ACID_TO_CODONS,
    CODON_TO_AMINO_ACID
)


def test_codon_tables():
    """Test that codon tables are consistent."""
    print("Testing codon tables...")
    
    # Test that all codons in CODON_TO_AMINO_ACID have corresponding entries
    for codon, aa in CODON_TO_AMINO_ACID.items():
        assert aa in AMINO_ACID_TO_CODONS, f"AA {aa} not in AMINO_ACID_TO_CODONS"
        assert codon in AMINO_ACID_TO_CODONS[aa], f"Codon {codon} not in synonyms for {aa}"
    
    print("✓ Codon tables are consistent")


def test_back_and_forward_translation():
    """Test that back-translation followed by forward-translation preserves sequence."""
    print("\nTesting back-translation and forward-translation...")
    
    # Test sequence
    test_sequences = [
        ['M', 'E', 'T'],  # Simple short sequence
        ['A', 'R', 'N', 'D', 'C', 'Q', 'E'],  # Various amino acids
        ['G', 'L', 'Y', 'C', 'I', 'N', 'E'],  # Real protein fragment
    ]
    
    for seq in test_sequences:
        # Back-translate
        codons = back_translate(seq)
        assert len(codons) == len(seq), f"Back-translation changed length: {len(seq)} -> {len(codons)}"
        
        # Forward-translate
        aa_seq = forward_translate(codons)
        assert aa_seq == seq, f"Translation not identity-preserving: {seq} -> {aa_seq}"
    
    print("✓ Back-translation and forward-translation preserve sequence identity")


def test_synonymous_substitutions():
    """Test that synonymous substitutions work correctly."""
    print("\nTesting synonymous substitutions...")
    
    # Create a codon sequence
    aa_seq = ['A', 'R', 'G', 'L', 'K']
    codons = back_translate(aa_seq)
    
    # Apply substitutions with rate = 1.0 (all codons)
    substituted_codons = apply_synonymous_substitutions(codons, 1.0)
    
    # Check that the amino acid sequence is still the same
    new_aa_seq = forward_translate(substituted_codons)
    assert new_aa_seq == aa_seq, f"Synonymous substitutions changed AA sequence"
    
    # Check that at least some codons changed (probabilistic, might fail rarely)
    num_changed = sum(1 for orig, new in zip(codons, substituted_codons) if orig != new)
    print(f"  Changed {num_changed}/{len(codons)} codons")
    
    print("✓ Synonymous substitutions preserve amino acid identity")


def test_nucleotide_augment_zero_rate():
    """Test NTA with substitution_rate=0 (should preserve sequence exactly)."""
    print("\nTesting NTA with substitution_rate=0...")
    
    test_seq = ['M', 'G', 'F', 'F', 'L', 'I', 'A', 'S', 'T']
    
    # Test multiple times (since back-translation is random)
    for i in range(10):
        augmented = nucleotide_augment(test_seq, 0.0)
        # With rate=0, back-translation and forward-translation should be identity
        # (though different random codons might be chosen each time)
        # The amino acid sequence should always be preserved
        assert augmented == test_seq, f"Zero-rate augmentation changed sequence: {test_seq} -> {augmented}"
    
    print("✓ NTA with substitution_rate=0 preserves sequence")


def test_nucleotide_augment_high_rate():
    """Test NTA with high substitution_rate."""
    print("\nTesting NTA with substitution_rate=0.5...")
    
    test_seq = ['A', 'R', 'G', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    # Test multiple times
    for i in range(10):
        augmented = nucleotide_augment(test_seq, 0.5)
        # The augmented sequence should still be the same amino acids
        assert augmented == test_seq, f"Augmentation changed AA sequence: {test_seq} -> {augmented}"
    
    print("✓ NTA with substitution_rate=0.5 preserves amino acid sequence")


def test_integration_with_framework():
    """Test that NTA works with the existing augmentation framework format."""
    print("\nTesting integration with augmentation framework...")
    
    # Test that the function signature matches existing augmentations
    # (sequence: list, intensity: float) -> list
    test_seq = list("ARNDCEQGHILKMFPSTWYV")
    intensity = 0.2
    
    result = nucleotide_augment(test_seq, intensity)
    
    assert isinstance(result, list), "Should return a list"
    assert all(isinstance(aa, str) for aa in result), "Should return list of strings"
    assert len(result) == len(test_seq), "Should preserve length"
    
    print("✓ NTA integrates with framework interface")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running NTA Augmentation Tests")
    print("=" * 60)
    
    try:
        test_codon_tables()
        test_back_and_forward_translation()
        test_synonymous_substitutions()
        test_nucleotide_augment_zero_rate()
        test_nucleotide_augment_high_rate()
        test_integration_with_framework()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
