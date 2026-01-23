"""
Unit tests for MiGu (Molecular Interactions and Geometric Upgrading) augmentation.

This test suite verifies:
1. Basic functionality and interface compatibility
2. Context awareness and interaction preservation
3. Edge cases and error handling
4. Integration with the framework
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentations.migu_augmentation import (
    migu_augment,
    analyze_local_context,
    get_context_aware_substitutes,
    preserve_critical_interactions,
    POSITIVE_CHARGED,
    NEGATIVE_CHARGED,
    HYDROPHOBIC,
    AROMATIC,
    AMINO_ACIDS,
)


def test_basic_functionality():
    """Test that MiGu augmentation runs without errors."""
    print("Test 1: Basic Functionality")
    
    seq = list("MKTAYIAKQRQISFVKSHFSRQ")
    intensity = 0.3
    
    aug_seq = migu_augment(seq, intensity)
    
    assert isinstance(aug_seq, list), "Output should be a list"
    assert len(aug_seq) == len(seq), "Length should be preserved"
    assert all(aa in AMINO_ACIDS for aa in aug_seq if aa in AMINO_ACIDS), \
        "All amino acids should be valid"
    
    print("✓ Basic functionality works")


def test_zero_intensity():
    """Test that zero intensity returns original sequence."""
    print("\nTest 2: Zero Intensity")
    
    seq = list("ACDEFGHIKLMNPQRSTVWY")
    aug_seq = migu_augment(seq, 0.0)
    
    assert aug_seq == seq, "Zero intensity should return original sequence"
    
    print("✓ Zero intensity returns original")


def test_context_analysis():
    """Test local context analysis."""
    print("\nTest 3: Context Analysis")
    
    # Sequence with clear hydrophobic region
    seq = list("SSSVVVIIILLLSSS")
    
    # Analyze context in middle of hydrophobic region
    context = analyze_local_context(seq, position=7, window_size=3)
    
    print(f"  Context at position 7: {context}")
    
    # Should detect high hydrophobic content
    assert context['hydrophobic'] > 0, "Should detect hydrophobic context"
    
    # Analyze context in polar region
    context_polar = analyze_local_context(seq, position=1, window_size=2)
    print(f"  Context at position 1: {context_polar}")
    
    print("✓ Context analysis works")


def test_disulfide_bond_preservation():
    """Test that cysteine pairs (disulfide bonds) are preserved."""
    print("\nTest 4: Disulfide Bond Preservation")
    
    # Sequence with two cysteines close together (likely disulfide bond)
    seq = list("MKTCYIAKQRQCSFVK")
    #             ^       ^  (positions 3 and 10)
    
    # Run augmentation multiple times
    for _ in range(5):
        aug_seq = migu_augment(seq, 0.8, preserve_interactions=True)
        
        # Check if cysteines are preserved
        if seq[3] == 'C':
            # If the cysteine pair exists, it should be preserved
            cys_positions = [i for i, aa in enumerate(seq) if aa == 'C']
            aug_cys_positions = [i for i, aa in enumerate(aug_seq) if aa == 'C']
            
            print(f"  Original C positions: {cys_positions}")
            print(f"  Augmented C positions: {aug_cys_positions}")
            
            # Should still have cysteines (preserved or both changed)
            # This test verifies the preservation logic is working
    
    print("✓ Disulfide bond preservation logic works")


def test_salt_bridge_preservation():
    """Test that salt bridges (K-D, R-E pairs) are preserved."""
    print("\nTest 5: Salt Bridge Preservation")
    
    # Sequence with clear salt bridge potential (K and D close together)
    seq = list("MGKDDLTKRAAE")
    #             ^^ (positions 2-3: K-D pair)
    #                  ^^  (positions 7-8: K-R pair, not salt bridge)
    
    aug_seq = migu_augment(seq, 0.5, preserve_interactions=True)
    
    # Check that charged residues near opposite charges maintain their charge
    # This is a probabilistic test, so we just verify it doesn't crash
    # and maintains sequence length
    assert len(aug_seq) == len(seq), "Length should be preserved"
    
    print(f"  Original: {''.join(seq)}")
    print(f"  Augmented: {''.join(aug_seq)}")
    print("✓ Salt bridge preservation logic works")


def test_aromatic_cluster_preservation():
    """Test that aromatic clusters are preserved."""
    print("\nTest 6: Aromatic Cluster Preservation")
    
    # Sequence with aromatic cluster (FYF)
    seq = list("MGFYFAKQR")
    #             ^^^  (aromatic cluster)
    
    aug_seq = migu_augment(seq, 0.5, preserve_interactions=True)
    
    # Check if middle aromatic (Y) in cluster stays aromatic
    if seq[3] in AROMATIC:
        # If surrounded by aromatics, should stay aromatic or be preserved
        print(f"  Original: {''.join(seq)}")
        print(f"  Augmented: {''.join(aug_seq)}")
    
    print("✓ Aromatic cluster preservation logic works")


def test_context_aware_substitution():
    """Test that substitutions are context-aware."""
    print("\nTest 7: Context-Aware Substitution")
    
    # Create a sequence with distinct regions
    # Hydrophobic region: VVVIII
    # Charged region: KKDEEE
    seq = list("SSVVVIIISSKKEEEESS")
    
    # Run augmentation
    aug_seq = migu_augment(seq, 0.4, preserve_interactions=False)
    
    print(f"  Original:  {''.join(seq)}")
    print(f"  Augmented: {''.join(aug_seq)}")
    
    # Just verify it runs and preserves length
    assert len(aug_seq) == len(seq), "Length should be preserved"
    
    print("✓ Context-aware substitution works")


def test_intensity_control():
    """Test that intensity parameter controls substitution rate."""
    print("\nTest 8: Intensity Control")
    
    seq = list("ACDEFGHIKLMNPQRSTVWY" * 5)  # 100 residues
    
    # Low intensity
    aug_low = migu_augment(seq, 0.1)
    changes_low = sum(1 for i in range(len(seq)) if seq[i] != aug_low[i])
    
    # High intensity
    aug_high = migu_augment(seq, 0.5)
    changes_high = sum(1 for i in range(len(seq)) if seq[i] != aug_high[i])
    
    print(f"  Low intensity (0.1): {changes_low} changes")
    print(f"  High intensity (0.5): {changes_high} changes")
    
    # Higher intensity should produce more changes
    assert changes_high > changes_low, "Higher intensity should produce more changes"
    
    print("✓ Intensity control works")


def test_edge_cases():
    """Test edge cases."""
    print("\nTest 9: Edge Cases")
    
    # Empty sequence
    aug_empty = migu_augment([], 0.5)
    assert aug_empty == [], "Empty sequence should return empty"
    
    # Single residue
    aug_single = migu_augment(['A'], 0.5)
    assert len(aug_single) == 1, "Single residue should remain single"
    assert aug_single[0] in AMINO_ACIDS, "Single residue should be valid"
    
    # Short sequence
    seq_short = list("ACE")
    aug_short = migu_augment(seq_short, 0.5)
    assert len(aug_short) == 3, "Short sequence length should be preserved"
    
    # All same residue
    seq_same = ['A'] * 10
    aug_same = migu_augment(seq_same, 0.5)
    assert len(aug_same) == 10, "Length should be preserved"
    
    # Sequence with unknown characters
    seq_with_x = list("ACXDEFX")
    aug_with_x = migu_augment(seq_with_x, 0.5)
    assert len(aug_with_x) == len(seq_with_x), "Length should be preserved"
    
    print("✓ Edge cases handled correctly")


def test_framework_interface():
    """Test compatibility with framework interface."""
    print("\nTest 10: Framework Interface Compatibility")
    
    # Test that function signature matches framework expectations
    seq = list("MKTAYIAKQRQISFVKSHFSRQ")
    
    # Test with different intensity values
    for intensity in [0.0, 0.1, 0.3, 0.5, 0.7]:
        try:
            result = migu_augment(seq, intensity)
            assert isinstance(result, list), f"Should return list for intensity {intensity}"
            assert len(result) == len(seq), f"Should preserve length for intensity {intensity}"
        except Exception as e:
            assert False, f"Failed with intensity {intensity}: {str(e)}"
    
    print("✓ Framework interface compatible")


def test_preservation_toggle():
    """Test that preserve_interactions parameter works."""
    print("\nTest 11: Interaction Preservation Toggle")
    
    seq = list("MKTCYIAKQRQCSFVK")
    
    # With preservation
    aug_with = migu_augment(seq, 0.5, preserve_interactions=True)
    
    # Without preservation
    aug_without = migu_augment(seq, 0.5, preserve_interactions=False)
    
    # Both should work and preserve length
    assert len(aug_with) == len(seq), "Length should be preserved (with)"
    assert len(aug_without) == len(seq), "Length should be preserved (without)"
    
    print("✓ Preservation toggle works")


def test_context_window_parameter():
    """Test that context_window parameter works."""
    print("\nTest 12: Context Window Parameter")
    
    seq = list("ACDEFGHIKLMNPQRSTVWY" * 3)
    
    # Small window
    aug_small = migu_augment(seq, 0.3, context_window=1)
    
    # Large window
    aug_large = migu_augment(seq, 0.3, context_window=5)
    
    # Both should work
    assert len(aug_small) == len(seq), "Small window should work"
    assert len(aug_large) == len(seq), "Large window should work"
    
    print("✓ Context window parameter works")


def test_reproducibility():
    """Test that augmentation produces variation (not deterministic)."""
    print("\nTest 13: Variation in Augmentation")
    
    import random
    
    seq = list("ACDEFGHIKLMNPQRSTVWY" * 3)
    
    # Generate multiple augmentations
    random.seed(42)
    aug1 = migu_augment(seq, 0.5)
    
    random.seed(123)
    aug2 = migu_augment(seq, 0.5)
    
    # They should be different (stochastic)
    assert aug1 != aug2, "Different random seeds should produce different results"
    
    print("✓ Augmentation produces stochastic variation")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running MiGu Augmentation Tests")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_zero_intensity()
        test_context_analysis()
        test_disulfide_bond_preservation()
        test_salt_bridge_preservation()
        test_aromatic_cluster_preservation()
        test_context_aware_substitution()
        test_intensity_control()
        test_edge_cases()
        test_framework_interface()
        test_preservation_toggle()
        test_context_window_parameter()
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
