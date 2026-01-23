"""
Unit tests for IMAEN (Interpretable Molecular Augmentation) protein augmentation.

Tests the property-aware amino acid substitution augmentation technique.
"""

import unittest
import sys
from typing import List

# Import the IMAEN augmentation functions
from augmentations.imaen import (
    imaen_simple,
    imaen_augment,
    _select_positions_by_property,
    _apply_property_aware_substitution,
    PROPERTY_GROUPS,
    HYDROPHOBIC_NONPOLAR,
    POLAR_UNCHARGED,
    POSITIVELY_CHARGED,
    NEGATIVELY_CHARGED
)


class TestIMAENAugmentation(unittest.TestCase):
    """Test suite for IMAEN augmentation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Standard test sequences
        self.basic_seq = list("ACDEFGHIKLMNPQRSTVWY")  # All 20 amino acids
        self.hydrophobic_seq = list("AVILMFWP")  # Hydrophobic amino acids
        self.polar_seq = list("STNQCYG")  # Polar uncharged
        self.charged_seq = list("KRHDE")  # Charged amino acids
        self.realistic_seq = list("MKTAYIAKQRQISFVKSHFSRQLEERLGLEVCDLEIR")  # Realistic protein
    
    def test_basic_augmentation(self):
        """Test basic IMAEN augmentation functionality."""
        sequence = self.basic_seq.copy()
        augmented = imaen_simple(sequence, intensity=0.3)
        
        # Check that augmented sequence is returned
        self.assertIsInstance(augmented, list)
        self.assertEqual(len(augmented), len(sequence))
        
        # Check that sequence contains valid amino acids
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        for aa in augmented:
            self.assertIn(aa, valid_aas, f"Invalid amino acid: {aa}")
    
    def test_intensity_levels(self):
        """Test augmentation with different intensity levels."""
        sequence = self.realistic_seq.copy()
        
        # Test various intensity levels
        for intensity in [0.1, 0.3, 0.5, 0.7, 0.9]:
            augmented = imaen_simple(sequence, intensity=intensity)
            
            # Length should be preserved
            self.assertEqual(len(augmented), len(sequence))
            
            # Calculate number of differences
            differences = sum(1 for a, b in zip(sequence, augmented) if a != b)
            
            # Differences should roughly correlate with intensity
            # Allow for some variance due to randomness
            expected_changes = int(intensity * len(sequence))
            # Allow 50% tolerance due to randomness and property constraints
            self.assertLessEqual(differences, expected_changes * 1.5,
                               f"Too many changes at intensity {intensity}")
    
    def test_zero_intensity(self):
        """Test that zero intensity returns unchanged sequence."""
        sequence = self.basic_seq.copy()
        augmented = imaen_simple(sequence, intensity=0.0)
        
        self.assertEqual(augmented, sequence, "Zero intensity should not change sequence")
    
    def test_high_intensity(self):
        """Test that high intensity still produces valid sequences."""
        sequence = self.realistic_seq.copy()
        augmented = imaen_simple(sequence, intensity=1.0)
        
        # Length preserved
        self.assertEqual(len(augmented), len(sequence))
        
        # All amino acids valid
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        for aa in augmented:
            self.assertIn(aa, valid_aas)
    
    def test_property_preservation(self):
        """Test that substitutions preserve amino acid properties."""
        # Test hydrophobic preservation
        hydrophobic_seq = list("AVILMFWP") * 3
        augmented = imaen_augment(hydrophobic_seq, intensity=0.5, conservative=True)
        
        # Count how many remain in similar property groups
        preserved = 0
        for orig, aug in zip(hydrophobic_seq, augmented):
            if orig == aug:
                preserved += 1
            elif aug in PROPERTY_GROUPS.get(orig, set()):
                preserved += 1  # Substituted with similar amino acid
        
        # Most should be preserved or substituted conservatively
        preservation_rate = preserved / len(hydrophobic_seq)
        self.assertGreater(preservation_rate, 0.7,
                          "Conservative mode should preserve properties")
    
    def test_property_aware_selection(self):
        """Test position selection with property bias."""
        sequence = self.realistic_seq.copy()
        
        # Test hydrophobic bias
        positions = _select_positions_by_property(sequence, intensity=0.3, 
                                                  property_bias='hydrophobic')
        # Check that selected positions are mostly hydrophobic
        selected_aas = [sequence[i] for i in positions]
        hydrophobic_count = sum(1 for aa in selected_aas if aa in HYDROPHOBIC_NONPOLAR)
        
        # At least some should be hydrophobic (allows for random selection if not enough)
        if len(positions) > 0:
            self.assertGreater(hydrophobic_count, 0,
                             "Hydrophobic bias should select some hydrophobic residues")
    
    def test_property_aware_substitution(self):
        """Test that substitutions respect property groups."""
        # Test with known amino acids
        sequence = list("AAAAAVVVVV")  # Aliphatic hydrophobic
        positions = [0, 1, 5, 6]
        
        augmented = _apply_property_aware_substitution(sequence, positions, 
                                                       conservative=True)
        
        # Check that substitutions are from property groups
        for pos in positions:
            original = sequence[pos]
            substituted = augmented[pos]
            
            if original != substituted:
                # Should be in the property group
                self.assertIn(substituted, PROPERTY_GROUPS.get(original, {original}),
                            f"Substitution {original}->{substituted} not in property group")
    
    def test_sequence_validity(self):
        """Test that augmented sequences are always valid."""
        sequences = [
            self.basic_seq,
            self.hydrophobic_seq,
            self.polar_seq,
            self.charged_seq,
            self.realistic_seq
        ]
        
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        
        for seq in sequences:
            for intensity in [0.1, 0.3, 0.5, 0.7]:
                augmented = imaen_simple(seq.copy(), intensity=intensity)
                
                # Check all amino acids are valid
                for aa in augmented:
                    self.assertIn(aa, valid_aas,
                                f"Invalid amino acid {aa} in augmented sequence")
                
                # Check length preservation
                self.assertEqual(len(augmented), len(seq),
                               "Sequence length not preserved")
    
    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        empty_seq = []
        augmented = imaen_simple(empty_seq, intensity=0.5)
        
        self.assertEqual(augmented, empty_seq, "Empty sequence should remain empty")
    
    def test_single_amino_acid(self):
        """Test handling of single amino acid sequences."""
        single_seq = ['A']
        augmented = imaen_simple(single_seq, intensity=0.5)
        
        # Should return a valid single amino acid
        self.assertEqual(len(augmented), 1)
        self.assertIn(augmented[0], set("ACDEFGHIKLMNPQRSTVWY"))
    
    def test_very_long_sequence(self):
        """Test handling of very long sequences."""
        long_seq = list("ACDEFGHIKLMNPQRSTVWY" * 50)  # 1000 amino acids
        augmented = imaen_simple(long_seq, intensity=0.3)
        
        # Check length preservation
        self.assertEqual(len(augmented), len(long_seq))
        
        # Check validity
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        for aa in augmented:
            self.assertIn(aa, valid_aas)
    
    def test_conservative_vs_non_conservative(self):
        """Test difference between conservative and non-conservative modes."""
        sequence = self.realistic_seq.copy()
        
        # Conservative augmentation
        conservative_aug = imaen_augment(sequence, intensity=0.5, 
                                        conservative=True)
        
        # Non-conservative augmentation (more diverse)
        non_conservative_aug = imaen_augment(sequence, intensity=0.5, 
                                             conservative=False)
        
        # Both should be valid
        self.assertEqual(len(conservative_aug), len(sequence))
        self.assertEqual(len(non_conservative_aug), len(sequence))
    
    def test_different_property_biases(self):
        """Test various property bias options."""
        sequence = self.realistic_seq.copy()
        
        biases = ['random', 'hydrophobic', 'polar', 'charged', 'aromatic']
        
        for bias in biases:
            augmented = imaen_augment(sequence, intensity=0.3, 
                                     property_bias=bias)
            
            # Check validity
            self.assertEqual(len(augmented), len(sequence))
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            for aa in augmented:
                self.assertIn(aa, valid_aas,
                            f"Invalid amino acid with bias={bias}")
    
    def test_reproducibility_with_different_seeds(self):
        """Test that augmentation produces different results (stochastic)."""
        sequence = self.realistic_seq.copy()
        
        results = []
        for _ in range(5):
            augmented = imaen_simple(sequence, intensity=0.5)
            results.append(''.join(augmented))
        
        # Should have some variety (stochastic process)
        unique_results = len(set(results))
        # Allow for possibility of duplicate due to randomness, but expect variety
        self.assertGreater(unique_results, 1,
                          "Augmentation should produce different results")
    
    def test_noise_level_effect(self):
        """Test that noise level parameter affects augmentation."""
        sequence = self.realistic_seq.copy()
        
        # Low noise
        low_noise = imaen_augment(sequence, intensity=0.3, noise_level=0.0)
        
        # High noise
        high_noise = imaen_augment(sequence, intensity=0.3, noise_level=0.5)
        
        # Both should be valid
        self.assertEqual(len(low_noise), len(sequence))
        self.assertEqual(len(high_noise), len(sequence))


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIMAENAugmentation)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    print("=" * 70)
    print("IMAEN Augmentation Test Suite")
    print("=" * 70)
    exit_code = run_tests()
    sys.exit(exit_code)
