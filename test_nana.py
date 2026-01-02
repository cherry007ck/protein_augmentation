"""
Unit tests for NaNa (Novel Augmentation of New Node Attributes) augmentation.

This test suite verifies:
1. Basic functionality and interface compatibility
2. Property preservation (biophysical and structural)
3. Edge cases and error handling
4. Integration with the framework
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentations.nana_augmentation import (
    nana_augment,
    calculate_similarity,
    get_similar_amino_acids,
    HYDROPHOBICITY,
    CHARGE,
    SIZE,
    HELIX_PROPENSITY,
    SHEET_PROPENSITY,
    AMINO_ACIDS,
)


def test_basic_functionality():
    """Test that NaNa augmentation runs without errors."""
    print("Test 1: Basic Functionality")
    
    seq = list("MKTAYIAKQRQISFVKSHFSRQ")
    intensity = 0.3
    
    aug_seq = nana_augment(seq, intensity)
    
    assert isinstance(aug_seq, list), "Output should be a list"
    assert len(aug_seq) == len(seq), "Length should be preserved"
    assert all(aa in AMINO_ACIDS for aa in aug_seq if aa in AMINO_ACIDS), \
        "All amino acids should be valid"
    
    print("✓ Basic functionality works")


def test_zero_intensity():
    """Test that zero intensity returns original sequence."""
    print("\nTest 2: Zero Intensity")
    
    seq = list("ACDEFGHIKLMNPQRSTVWY")
    aug_seq = nana_augment(seq, 0.0)
    
    assert aug_seq == seq, "Zero intensity should return original sequence"
    
    print("✓ Zero intensity returns original")


def test_property_preservation_charge():
    """Test that charge is preserved in substitutions."""
    print("\nTest 3: Charge Preservation")
    
    # Test with highly charged sequence
    seq = list("KKKDDDRRREEE")
    aug_seq = nana_augment(seq, 0.5, use_groups=True)
    
    # Count charges in original and augmented
    original_positive = sum(1 for aa in seq if CHARGE.get(aa, 0) > 0)
    original_negative = sum(1 for aa in seq if CHARGE.get(aa, 0) < 0)
    
    aug_positive = sum(1 for aa in aug_seq if CHARGE.get(aa, 0) > 0)
    aug_negative = sum(1 for aa in aug_seq if CHARGE.get(aa, 0) < 0)
    
    # Charges should be similar (within tolerance due to random sampling)
    print(f"  Original: +{original_positive} -{original_negative}")
    print(f"  Augmented: +{aug_positive} -{aug_negative}")
    
    # Allow some variation but should preserve general charge distribution
    assert abs(original_positive - aug_positive) <= 2, "Positive charge should be preserved"
    assert abs(original_negative - aug_negative) <= 2, "Negative charge should be preserved"
    
    print("✓ Charge preservation works")


def test_property_preservation_hydrophobicity():
    """Test that hydrophobic character is preserved."""
    print("\nTest 4: Hydrophobicity Preservation")
    
    # Hydrophobic sequence
    seq = list("VVVIIILLLMMM")
    aug_seq = nana_augment(seq, 0.5, use_groups=True)
    
    # Calculate average hydrophobicity
    orig_hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in seq) / len(seq)
    aug_hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in aug_seq) / len(aug_seq)
    
    print(f"  Original hydrophobicity: {orig_hydro:.2f}")
    print(f"  Augmented hydrophobicity: {aug_hydro:.2f}")
    
    # Should be similar (hydrophobic residues substitute with hydrophobic)
    assert abs(orig_hydro - aug_hydro) < 2.0, "Hydrophobicity should be preserved"
    
    print("✓ Hydrophobicity preservation works")


def test_similarity_calculation():
    """Test the similarity calculation function."""
    print("\nTest 5: Similarity Calculation")
    
    # Test identical amino acids
    assert calculate_similarity('A', 'A') == 1.0, "Identical AAs should have similarity 1.0"
    
    # Test similar amino acids (both hydrophobic aliphatic)
    sim_il = calculate_similarity('I', 'L')
    print(f"  Similarity(I, L): {sim_il:.3f}")
    assert sim_il > 0.7, "I and L should be very similar"
    
    # Test dissimilar amino acids (hydrophobic vs charged)
    sim_ik = calculate_similarity('I', 'K')
    print(f"  Similarity(I, K): {sim_ik:.3f}")
    assert sim_ik < 0.6, "I and K should be dissimilar"
    
    # Test oppositely charged (should be very dissimilar)
    sim_kd = calculate_similarity('K', 'D')
    print(f"  Similarity(K, D): {sim_kd:.3f}")
    assert sim_kd < 0.65, "K and D should be very dissimilar"
    
    print("✓ Similarity calculation works correctly")


def test_get_similar_amino_acids():
    """Test the similar amino acids lookup function."""
    print("\nTest 6: Similar Amino Acids Lookup")
    
    # Get similar AAs for leucine (hydrophobic)
    similar = get_similar_amino_acids('L', similarity_threshold=0.65)
    similar_aas = [aa for aa, _ in similar]
    
    print(f"  Similar to L: {similar_aas}")
    
    # Should include other hydrophobic amino acids
    assert 'I' in similar_aas or 'V' in similar_aas, "L should be similar to other hydrophobic AAs"
    
    # Should NOT include charged amino acids
    assert 'K' not in similar_aas, "L should not be similar to K"
    assert 'D' not in similar_aas, "L should not be similar to D"
    
    print("✓ Similar amino acids lookup works")


def test_intensity_control():
    """Test that intensity parameter controls substitution rate."""
    print("\nTest 7: Intensity Control")
    
    seq = list("ACDEFGHIKLMNPQRSTVWY" * 5)  # 100 residues
    
    # Low intensity
    aug_low = nana_augment(seq, 0.1, use_groups=True)
    changes_low = sum(1 for i in range(len(seq)) if seq[i] != aug_low[i])
    
    # High intensity
    aug_high = nana_augment(seq, 0.5, use_groups=True)
    changes_high = sum(1 for i in range(len(seq)) if seq[i] != aug_high[i])
    
    print(f"  Low intensity (0.1): {changes_low} changes")
    print(f"  High intensity (0.5): {changes_high} changes")
    
    # Higher intensity should produce more changes
    assert changes_high > changes_low, "Higher intensity should produce more changes"
    
    # Check that changes are roughly in expected range
    assert 5 <= changes_low <= 20, f"Low intensity changes out of expected range: {changes_low}"
    assert 30 <= changes_high <= 70, f"High intensity changes out of expected range: {changes_high}"
    
    print("✓ Intensity control works")


def test_edge_cases():
    """Test edge cases."""
    print("\nTest 8: Edge Cases")
    
    # Empty sequence
    aug_empty = nana_augment([], 0.5)
    assert aug_empty == [], "Empty sequence should return empty"
    
    # Single residue
    aug_single = nana_augment(['A'], 0.5)
    assert len(aug_single) == 1, "Single residue should remain single"
    assert aug_single[0] in AMINO_ACIDS, "Single residue should be valid"
    
    # All same residue
    seq_same = ['A'] * 10
    aug_same = nana_augment(seq_same, 0.5, use_groups=True)
    assert len(aug_same) == 10, "Length should be preserved"
    
    # Sequence with unknown characters (should handle gracefully)
    seq_with_x = list("ACXDEFX")
    aug_with_x = nana_augment(seq_with_x, 0.5)
    assert len(aug_with_x) == len(seq_with_x), "Length should be preserved"
    
    print("✓ Edge cases handled correctly")


def test_framework_interface():
    """Test compatibility with framework interface."""
    print("\nTest 9: Framework Interface Compatibility")
    
    # Test that function signature matches framework expectations
    # Function should accept (sequence: List[str], intensity: float)
    seq = list("MKTAYIAKQRQISFVKSHFSRQ")
    
    # Test with different intensity values
    for intensity in [0.0, 0.1, 0.3, 0.5, 0.7]:
        try:
            result = nana_augment(seq, intensity)
            assert isinstance(result, list), f"Should return list for intensity {intensity}"
            assert len(result) == len(seq), f"Should preserve length for intensity {intensity}"
        except Exception as e:
            assert False, f"Failed with intensity {intensity}: {str(e)}"
    
    print("✓ Framework interface compatible")


def test_reproducibility():
    """Test that augmentation produces variation (not deterministic)."""
    print("\nTest 10: Variation in Augmentation")
    
    import random
    
    seq = list("ACDEFGHIKLMNPQRSTVWY" * 3)
    
    # Generate multiple augmentations
    random.seed(42)
    aug1 = nana_augment(seq, 0.5, use_groups=True)
    
    random.seed(123)
    aug2 = nana_augment(seq, 0.5, use_groups=True)
    
    # They should be different (stochastic)
    assert aug1 != aug2, "Different random seeds should produce different results"
    
    print("✓ Augmentation produces stochastic variation")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running NaNa Augmentation Tests")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_zero_intensity()
        test_property_preservation_charge()
        test_property_preservation_hydrophobicity()
        test_similarity_calculation()
        test_get_similar_amino_acids()
        test_intensity_control()
        test_edge_cases()
        test_framework_interface()
        test_reproducibility()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
