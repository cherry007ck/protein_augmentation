"""
Test script for Spider augmentation implementation.

This script verifies that the Spider augmentation:
1. Performs random amino acid substitutions correctly
2. Performs random amino acid insertions correctly
3. Works with various intensity values
4. Integrates properly with the existing augmentation framework
5. Returns valid amino acid sequences
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentations.spider_augmentation import (
    spider_augment,
    random_substitute_residues,
    random_insert_residues,
    AMINO_ACIDS
)


def test_spider_basic_functionality():
    """Test basic Spider augmentation functionality."""
    print("Testing Spider basic functionality...")
    
    test_seq = ['M', 'E', 'T', 'H', 'Y', 'L', 'A', 'M', 'I', 'N', 'E']
    
    # Test with various intensities
    for intensity in [0.0, 0.2, 0.5, 0.8]:
        result = spider_augment(test_seq, intensity)
        assert isinstance(result, list), f"Should return a list"
        assert all(isinstance(aa, str) for aa in result), "Should return list of strings"
        assert all(aa in AMINO_ACIDS for aa in result), f"All AAs should be valid at intensity {intensity}"
    
    print("[PASS] Spider basic functionality works correctly")


def test_spider_zero_intensity():
    """Test Spider with intensity=0 (should preserve sequence)."""
    print("\nTesting Spider with intensity=0...")
    
    test_seq = ['M', 'G', 'F', 'F', 'L', 'I', 'A', 'S', 'T']
    
    # Test multiple times
    for i in range(10):
        result = spider_augment(test_seq, 0.0)
        assert result == test_seq, f"Zero intensity should preserve sequence"
    
    print("[PASS] Spider with intensity=0 preserves sequence")


def test_spider_substitutions():
    """Test the substitution component of Spider."""
    print("\nTesting Spider substitution component...")
    
    test_seq = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']  # All same AA
    
    # Apply only substitutions (substitution_rate > 0)
    result = random_substitute_residues(test_seq, 0.5)
    
    # Check that sequence length is preserved
    assert len(result) == len(test_seq), "Substitutions should preserve length"
    
    # Check that all are valid amino acids
    assert all(aa in AMINO_ACIDS for aa in result), "All substituted AAs should be valid"
    
    # Check that some changes occurred (probabilistic, might rarely fail)
    num_changed = sum(1 for a, b in zip(test_seq, result) if a != b)
    assert num_changed > 0, "Some substitutions should have occurred"
    
    print(f"  Changed {num_changed}/{len(test_seq)} residues")
    print("[PASS] Spider substitution component works correctly")


def test_spider_insertions():
    """Test the insertion component of Spider."""
    print("\nTesting Spider insertion component...")
    
    test_seq = ['M', 'E', 'T', 'H', 'Y', 'L']
    original_len = len(test_seq)
    
    # Apply insertions with rate 0.5 (should insert ~3 AAs for length 6)
    result = random_insert_residues(test_seq, 0.5)
    
    # Check that all are valid amino acids
    assert all(aa in AMINO_ACIDS for aa in result), "All inserted AAs should be valid"
    
    # Check that length increased
    assert len(result) >= original_len, "Insertions should increase or maintain length"
    
    expected_insertions = int(0.5 * original_len)
    print(f"  Original length: {original_len}, Result length: {len(result)}")
    print(f"  Expected ~{expected_insertions} insertions, got {len(result) - original_len}")
    print("[PASS] Spider insertion component works correctly")


def test_spider_returns_valid_amino_acids():
    """Test that Spider always returns valid amino acids."""
    print("\nTesting Spider returns only valid amino acids...")
    
    test_seq = list("ARNDCEQGHILKMFPSTWYV")  # All 20 standard AAs
    
    # Test with maximum intensity
    for _ in range(10):
        result = spider_augment(test_seq, 1.0)
        
        # Check all are valid
        for aa in result:
            assert aa in AMINO_ACIDS, f"Invalid amino acid found: {aa}"
    
    print("[PASS] Spider always returns valid amino acids")


def test_spider_integration_with_framework():
    """Test that Spider works with the existing augmentation framework format."""
    print("\nTesting integration with augmentation framework...")
    
    # Test that the function signature matches existing augmentations
    # (sequence: list, intensity: float) -> list
    test_seq = list("METHYLAMINE")
    intensity = 0.3
    
    result = spider_augment(test_seq, intensity)
    
    assert isinstance(result, list), "Should return a list"
    assert all(isinstance(aa, str) for aa in result), "Should return list of strings"
    assert all(len(aa) == 1 for aa in result), "Each element should be single character"
    
    print("[PASS] Spider integrates with framework interface")


def test_spider_length_changes():
    """Test that Spider appropriately changes sequence length."""
    print("\nTesting Spider sequence length changes...")
    
    test_seq = ['M', 'E', 'T', 'H', 'Y', 'L', 'A', 'M', 'I', 'N', 'E']
    original_len = len(test_seq)
    
    # With low intensity, length should not increase much
    result_low = spider_augment(test_seq, 0.1)
    
    # With high intensity, length should increase more
    result_high = spider_augment(test_seq, 0.8)
    
    print(f"  Original length: {original_len}")
    print(f"  Low intensity (0.1) length: {len(result_low)}")
    print(f"  High intensity (0.8) length: {len(result_high)}")
    
    # High intensity should generally result in longer sequences than low
    # (though this is probabilistic)
    assert len(result_high) >= len(result_low), \
        "Higher intensity should generally produce longer sequences"
    
    print("[PASS] Spider length changes work as expected")


def test_spider_with_edge_cases():
    """Test Spider with edge cases."""
    print("\nTesting Spider with edge cases...")
    
    # Empty sequence
    empty_seq = []
    result = spider_augment(empty_seq, 0.5)
    assert result == [], "Empty sequence should return empty"
    
    # Single amino acid
    single_seq = ['M']
    result = spider_augment(single_seq, 0.5)
    assert len(result) >= 1, "Single AA sequence should return valid result"
    assert all(aa in AMINO_ACIDS for aa in result), "Result should be valid"
    
    # Very long sequence
    long_seq = list("ARNDCEQGHILKMFPSTWYV" * 10)  # 200 AAs
    result = spider_augment(long_seq, 0.2)
    assert isinstance(result, list), "Long sequence should work"
    assert all(aa in AMINO_ACIDS for aa in result), "All AAs should be valid"
    
    # Intensity edge cases
    result = spider_augment(['M', 'E', 'T'], -0.5)  # Negative intensity
    assert isinstance(result, list), "Should handle negative intensity"
    
    result = spider_augment(['M', 'E', 'T'], 2.0)  # >1.0 intensity
    assert isinstance(result, list), "Should handle intensity > 1.0"
    
    print("[PASS] Spider handles edge cases correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Spider Augmentation Tests")
    print("=" * 60)
    
    try:
        test_spider_basic_functionality()
        test_spider_zero_intensity()
        test_spider_substitutions()
        test_spider_insertions()
        test_spider_returns_valid_amino_acids()
        test_spider_integration_with_framework()
        test_spider_length_changes()
        test_spider_with_edge_cases()
        
        print("\n" + "=" * 60)
        print("[PASS] All tests passed!")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
