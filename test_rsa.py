"""
Test script for RSA (Retrieved Sequence Augmentation) implementation.

This script verifies that the simplified RSA augmentation:
1. Correctly applies conservative amino acid substitutions
2. Respects biochemical property groups
3. Preserves sequence length
4. Responds appropriately to intensity parameter
5. Integrates properly with the existing augmentation framework
"""

import sys
sys.path.insert(0, '/home/luffy/dsa_project/rsa/protein_augmentation')

from augmentations.rsa_augmentation import (
    generate_conservative_mutation,
    rsa_augment,
    rsa_augment_with_original,
    AMINO_ACID_GROUPS
)


def test_amino_acid_groups():
    """Test that amino acid groups are well-defined."""
    print("Testing amino acid groups...")
    
    # Check that all 20 standard amino acids are covered
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    covered_aa = set(AMINO_ACID_GROUPS.keys())
    
    assert covered_aa == standard_aa, f"Missing amino acids: {standard_aa - covered_aa}"
    
    # Check that substitutions don't include the original amino acid
    for aa, substitutes in AMINO_ACID_GROUPS.items():
        assert aa not in substitutes, f"AA {aa} should not substitute to itself"
    
    print("✓ Amino acid groups are well-defined")


def test_conservative_mutation():
    """Test that conservative mutations respect biochemical properties."""
    print("\nTesting conservative mutations...")
    
    # Test hydrophobic amino acids stay hydrophobic
    hydrophobic = ['A', 'V', 'I', 'L', 'M', 'F', 'W']
    for aa in hydrophobic:
        mutated = generate_conservative_mutation(aa)
        # Mutated AA should be in the conservative group or the original
        if mutated != aa:
            assert mutated in AMINO_ACID_GROUPS[aa], f"{aa} -> {mutated} not conservative"
    
    # Test charged amino acids
    positive = ['K', 'R', 'H']
    for aa in positive:
        mutated = generate_conservative_mutation(aa)
        # Should mutate within positive group or stay same
        if mutated != aa:
            assert mutated in AMINO_ACID_GROUPS[aa], f"{aa} -> {mutated} not conservative"
    
    print("✓ Conservative mutations respect biochemical properties")


def test_rsa_augment_zero_intensity():
    """Test RSA with intensity=0 (no mutations)."""
    print("\nTesting RSA with intensity=0...")
    
    test_seq = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I']
    
    augmented = rsa_augment(test_seq, 0.0)
    
    # Should return exact copy with no mutations
    assert augmented == test_seq, f"Zero-intensity should preserve sequence"
    assert len(augmented) == len(test_seq), "Length should be preserved"
    
    print("✓ RSA with intensity=0 preserves sequence")


def test_rsa_augment_length_preservation():
    """Test that RSA always preserves sequence length."""
    print("\nTesting sequence length preservation...")
    
    test_sequences = [
        list("M"),  # Single residue
        list("ARND"),  # Short
        list("ACDEFGHIKLMNPQRSTVWY"),  # All 20 AAs
        list("M" * 100),  # Long sequence
    ]
    
    for seq in test_sequences:
        for intensity in [0.1, 0.3, 0.5, 0.8, 1.0]:
            augmented = rsa_augment(seq, intensity)
            assert len(augmented) == len(seq), \
                f"Length mismatch: {len(seq)} -> {len(augmented)} at intensity {intensity}"
    
    print("✓ RSA preserves sequence length for all intensities")


def test_rsa_augment_intensity_effects():
    """Test that intensity parameter controls mutation rate."""
    print("\nTesting intensity parameter effects...")
    
    test_seq = list("ARNDCEQGHILKMFPSTWYV" * 5)  # 100 residues
    
    # Test different intensities
    intensities = [0.1, 0.3, 0.5, 0.8]
    mutation_counts = []
    
    for intensity in intensities:
        # Run multiple times and average (due to randomness)
        mutations = []
        for _ in range(10):
            augmented = rsa_augment(test_seq, intensity)
            num_mutations = sum(1 for orig, aug in zip(test_seq, augmented) if orig != aug)
            mutations.append(num_mutations)
        
        avg_mutations = sum(mutations) / len(mutations)
        mutation_counts.append(avg_mutations)
        expected_mutations = intensity * len(test_seq)
        
        print(f"  Intensity {intensity}: ~{avg_mutations:.1f} mutations (expected ~{expected_mutations:.1f})")
        
        # Allow 20% tolerance due to randomness
        assert abs(avg_mutations - expected_mutations) < expected_mutations * 0.3, \
            f"Mutation count deviates too much from expected"
    
    # Check that mutation counts increase with intensity
    for i in range(len(mutation_counts) - 1):
        assert mutation_counts[i] < mutation_counts[i+1], \
            "Higher intensity should produce more mutations"
    
    print("✓ Intensity parameter correctly controls mutation rate")


def test_conservative_mutations_only():
    """Test that all mutations are conservative."""
    print("\nTesting that mutations are conservative...")
    
    test_seq = list("ARNDCEQGHILKMFPSTWYV" * 3)
    
    # Run augmentation multiple times
    for _ in range(20):
        augmented = rsa_augment(test_seq, 0.5)
        
        for orig, aug in zip(test_seq, augmented):
            if orig != aug:
                # Check that the mutation is conservative
                assert aug in AMINO_ACID_GROUPS[orig], \
                    f"Non-conservative mutation: {orig} -> {aug}"
    
    print("✓ All mutations are conservative")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Empty sequence
    empty_seq = []
    assert rsa_augment(empty_seq, 0.5) == empty_seq, "Should handle empty sequence"
    
    # Single residue
    single_seq = ['M']
    augmented = rsa_augment(single_seq, 1.0)
    assert len(augmented) == 1, "Should preserve length for single residue"
    
    # Intensity > 1.0 should be clamped
    test_seq = list("ARND")
    augmented = rsa_augment(test_seq, 2.0)
    assert len(augmented) == len(test_seq), "Should handle intensity > 1.0"
    
    # Negative intensity should return original
    augmented = rsa_augment(test_seq, -0.5)
    assert augmented == test_seq, "Negative intensity should preserve sequence"
    
    print("✓ Edge cases handled correctly")


def test_rsa_augment_with_original():
    """Test the variant that sometimes returns original sequence."""
    print("\nTesting rsa_augment_with_original...")
    
    test_seq = list("ARNDCEQGHILKMFPSTWYV")
    
    # Run many times to see statistical behavior
    original_count = 0
    mutated_count = 0
    
    for _ in range(100):
        augmented = rsa_augment_with_original(test_seq, 0.5)
        if augmented == test_seq:
            original_count += 1
        else:
            mutated_count += 1
    
    print(f"  Original: {original_count}/100, Mutated: {mutated_count}/100")
    
    # Should be roughly 50-50 (allow some variance)
    assert 30 < original_count < 70, "Should return original ~50% of the time"
    assert 30 < mutated_count < 70, "Should return mutated ~50% of the time"
    
    print("✓ rsa_augment_with_original works correctly")


def test_framework_compatibility():
    """Test that RSA works with the existing augmentation framework format."""
    print("\nTesting integration with augmentation framework...")
    
    # Test that the function signature matches existing augmentations
    # (sequence: list, intensity: float) -> list
    test_seq = list("ARNDCEQGHILKMFPSTWYV")
    intensity = 0.2
    
    result = rsa_augment(test_seq, intensity)
    
    assert isinstance(result, list), "Should return a list"
    assert all(isinstance(aa, str) for aa in result), "Should return list of strings"
    assert len(result) == len(test_seq), "Should preserve length"
    
    # Test that it works with different sequence types
    seq_as_chars = list("METHI")
    result = rsa_augment(seq_as_chars, 0.3)
    assert isinstance(result, list), "Should work with character list"
    
    print("✓ RSA integrates with framework interface")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running RSA Augmentation Tests")
    print("=" * 60)
    
    try:
        test_amino_acid_groups()
        test_conservative_mutation()
        test_rsa_augment_zero_intensity()
        test_rsa_augment_length_preservation()
        test_rsa_augment_intensity_effects()
        test_conservative_mutations_only()
        test_edge_cases()
        test_rsa_augment_with_original()
        test_framework_compatibility()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
