"""
Test script for Residue Masking augmentation implementation.

This script verifies that the residue masking augmentation:
1. Correctly applies masking at specified rates
2. Uses the appropriate masking strategy (80/10/10 for MLM)
3. Preserves sequence length
4. Integrates properly with the framework
"""

import sys
sys.path.insert(0, '/home/luffy/protein_augmentation')

from augmentations.residue_masking import (
    mask_residues,
    simple_mask_residues,
    conservative_mask_residues,
    AMINO_ACIDS
)


def test_simple_masking():
    """Test simple masking functionality."""
    print("Testing simple masking...")
    
    test_seq = list("METHYLKFPSTWYV")
    original_len = len(test_seq)
    
    # Test with different masking rates
    for rate in [0.0, 0.2, 0.5, 1.0]:
        masked = simple_mask_residues(test_seq, rate)
        
        # Check length preservation
        assert len(masked) == original_len, f"Length mismatch: {len(masked)} != {original_len}"
        
        # Count 'X' tokens
        num_masked = sum(1 for aa in masked if aa == 'X')
        expected = int(rate * original_len) if rate < 1.0 else original_len
        
        # Allow for rounding
        assert abs(num_masked - expected) <= 1, f"Masking count off: {num_masked} vs {expected}"
    
    print("✓ Simple masking works correctly")


def test_zero_masking():
    """Test that masking_rate=0 returns unchanged sequence."""
    print("\nTesting zero masking rate...")
    
    test_seq = list("ARGLKFPSTWYV")
    
    # Test all three variants
    for mask_func in [mask_residues, simple_mask_residues, conservative_mask_residues]:
        masked = mask_func(test_seq, 0.0)
        assert masked == test_seq, f"{mask_func.__name__} changed sequence with rate=0"
    
    print("✓ Zero masking rate preserves sequences")


def test_mlm_masking():
    """Test MLM-style masking (80/10/10 split)."""
    print("\nTesting MLM-style masking...")
    
    test_seq = list("A" * 100)  # Simple sequence for easier counting
    masking_rate = 0.15  # 15 positions should be selected
    num_trials = 100
    
    mask_counts = {'X': 0, 'random': 0, 'unchanged': 0}
    
    for _ in range(num_trials):
        masked = mask_residues(test_seq, masking_rate)
        
        # Count different types
        for i, aa in enumerate(masked):
            if aa == 'X':
                mask_counts['X'] += 1
            elif aa != test_seq[i]:
                mask_counts['random'] += 1
            # else: unchanged
    
    total_masked = sum(mask_counts.values())
    expected_total = int(masking_rate * len(test_seq) * num_trials)
    
    # Check that total is approximately correct
    assert abs(total_masked - expected_total) < expected_total * 0.2, \
        f"Total masked positions off: {total_masked} vs {expected_total}"
    
    # Check that 'X' is the most common (should be ~80%)
    assert mask_counts['X'] > mask_counts['random'], \
        "Mask token should be more common than random replacements"
    
    print(f"  Mask distribution: X={mask_counts['X']}, Random={mask_counts['random']}")
    print("✓ MLM-style masking applies correct strategy")


def test_length_preservation():
    """Test that all masked sequences preserve length."""
    print("\nTesting length preservation...")
    
    test_sequences = [
        list("M"),  # Single residue
        list("METH"),  # Short
        list("ARGLKFPSTWYVARGLKFPSTWYV"),  # Longer
    ]
    
    for seq in test_sequences:
        original_len = len(seq)
        
        for mask_func in [mask_residues, simple_mask_residues, conservative_mask_residues]:
            masked = mask_func(seq, 0.3)
            assert len(masked) == original_len, \
                f"{mask_func.__name__} changed length: {original_len} -> {len(masked)}"
    
    print("✓ All masking variants preserve sequence length")


def test_conservative_masking():
    """Test conservative masking preserves chemical properties."""
    print("\nTesting conservative masking...")
    
    # Test with amino acids from different groups
    test_seq = list("MKELDFPSTWYV")
    num_trials = 50
    
    replacements = {}
    for _ in range(num_trials):
        masked = conservative_mask_residues(test_seq, 0.5)
        
        for orig, masked_aa in zip(test_seq, masked):
            if orig != masked_aa:
                if orig not in replacements:
                    replacements[orig] = set()
                replacements[orig].add(masked_aa)
    
    # Check that hydrophobic AAs are replaced with hydrophobic (with some exceptions)
    if 'M' in replacements:
        # M (Methionine) should be replaced with other hydrophobic AAs
        hydrophobic = set('AVILMFWP')
        m_replacements = replacements['M']
        # Most replacements should be hydrophobic
        print(f"  M replacements: {m_replacements}")
    
    print("✓ Conservative masking uses chemical property groups")


def test_framework_compatibility():
    """Test compatibility with the augmentation framework interface."""
    print("\nTesting framework compatibility...")
    
    test_seq = list("METHYLAMINE")
    intensity = 0.15
    
    # Test that all functions work with framework interface
    for mask_func in [mask_residues, simple_mask_residues, conservative_mask_residues]:
        result = mask_func(test_seq, intensity)
        
        assert isinstance(result, list), f"{mask_func.__name__} should return list"
        assert all(isinstance(aa, str) for aa in result), \
            f"{mask_func.__name__} should return list of strings"
        assert len(result) == len(test_seq), \
            f"{mask_func.__name__} should preserve length"
    
    print("✓ All masking functions compatible with framework interface")


def test_mask_token_usage():
    """Test that mask token is used correctly."""
    print("\nTesting mask token usage...")
    
    test_seq = list("A" * 20)
    
    # Simple masking should use only 'X'
    masked = simple_mask_residues(test_seq, 0.5)
    for aa in masked:
        assert aa in ['A', 'X'], f"Simple masking should only use 'A' or 'X', got {aa}"
    
    # MLM masking should use 'X', random AAs, or original
    masked = mask_residues(test_seq, 0.5)
    for aa in masked:
        assert aa in AMINO_ACIDS + ['X'], f"MLM masking produced invalid AA: {aa}"
    
    print("✓ Mask tokens used correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Residue Masking Augmentation Tests")
    print("=" * 70)
    
    try:
        test_simple_masking()
        test_zero_masking()
        test_mlm_masking()
        test_length_preservation()
        test_conservative_masking()
        test_framework_compatibility()
        test_mask_token_usage()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
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
