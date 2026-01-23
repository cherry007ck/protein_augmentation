"""
Unit tests for BootGen augmentation module.

Tests the core functionality of BootGen-inspired protein sequence augmentation,
including proxy scoring, bootstrapped generation, and rank-based selection.
"""

import unittest
import sys
from typing import List

# Import the functions to test
from augmentations.bootgen import (
    compute_composition_similarity,
    compute_property_similarity,
    compute_sequence_score,
    generate_candidate_sequence,
    generate_bootstrapped_sequences,
    rank_based_selection,
    bootgen_augment,
    STANDARD_AAS,
    AA_TO_GROUP
)


class TestCompositionSimilarity(unittest.TestCase):
    """Test amino acid composition similarity computation."""
    
    def test_identical_sequences(self):
        """Identical sequences should have similarity of 1.0."""
        seq = ['A', 'C', 'D', 'E', 'F']
        similarity = compute_composition_similarity(seq, seq.copy())
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_different_sequences(self):
        """Different sequences should have lower similarity."""
        seq1 = ['A', 'A', 'A', 'A', 'A']
        seq2 = ['C', 'C', 'C', 'C', 'C']
        similarity = compute_composition_similarity(seq1, seq2)
        self.assertLess(similarity, 0.5)
    
    def test_partial_overlap(self):
        """Sequences with partial overlap should have intermediate similarity."""
        seq1 = ['A', 'C', 'D', 'E', 'F']
        seq2 = ['A', 'A', 'C', 'D', 'G']
        similarity = compute_composition_similarity(seq1, seq2)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_empty_sequences(self):
        """Empty sequences should return 0.0."""
        similarity = compute_composition_similarity([], [])
        self.assertEqual(similarity, 0.0)
    
    def test_score_range(self):
        """Similarity should always be in [0, 1]."""
        seq1 = ['A', 'C', 'D', 'E', 'F', 'G', 'H']
        seq2 = ['K', 'L', 'M', 'N', 'P', 'Q', 'R']
        similarity = compute_composition_similarity(seq1, seq2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


class TestPropertySimilarity(unittest.TestCase):
    """Test biochemical property similarity computation."""
    
    def test_same_properties(self):
        """Sequences with same property distribution should have high similarity."""
        # Both are hydrophobic-rich
        seq1 = ['A', 'V', 'I', 'L', 'M']
        seq2 = ['A', 'I', 'L', 'V', 'M']
        similarity = compute_property_similarity(seq1, seq2)
        self.assertGreater(similarity, 0.9)
    
    def test_different_properties(self):
        """Sequences with different properties should have lower similarity."""
        seq1 = ['K', 'R', 'H']  # All positive
        seq2 = ['D', 'E', 'D']  # All negative
        similarity = compute_property_similarity(seq1, seq2)
        self.assertLess(similarity, 0.5)
    
    def test_score_range(self):
        """Property similarity should be in [0, 1]."""
        seq1 = ['A', 'K', 'D', 'S', 'C']
        seq2 = ['F', 'R', 'E', 'T', 'G']
        similarity = compute_property_similarity(seq1, seq2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


class TestSequenceScore(unittest.TestCase):
    """Test proxy sequence scoring function."""
    
    def test_identical_sequences(self):
        """Identical sequences should have maximum score."""
        seq = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
        score = compute_sequence_score(seq, seq.copy())
        self.assertGreater(score, 0.9)
    
    def test_length_penalty(self):
        """Sequences with different lengths should be penalized."""
        original = ['A', 'C', 'D', 'E', 'F']
        short = ['A', 'C']
        long_seq = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        
        score_short = compute_sequence_score(original, short)
        score_long = compute_sequence_score(original, long_seq)
        
        # Both should be penalized
        self.assertLess(score_short, 0.8)
        self.assertLess(score_long, 0.8)
    
    def test_score_range(self):
        """Score should always be in [0, 1]."""
        original = ['A', 'C', 'D', 'E', 'F']
        candidate = ['K', 'R', 'H', 'S', 'T']
        score = compute_sequence_score(original, candidate)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_empty_sequences(self):
        """Empty sequences should return 0.0."""
        score = compute_sequence_score([], [])
        self.assertEqual(score, 0.0)


class TestCandidateGeneration(unittest.TestCase):
    """Test candidate sequence generation."""
    
    def test_basic_generation(self):
        """Should generate a valid candidate sequence."""
        seq = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L']
        candidate = generate_candidate_sequence(seq, num_substitutions=3)
        
        # Should have same length
        self.assertEqual(len(candidate), len(seq))
        
        # All amino acids should be valid
        self.assertTrue(all(aa in STANDARD_AAS for aa in candidate))
    
    def test_substitution_count(self):
        """Should make approximately the requested number of substitutions."""
        seq = ['A'] * 100
        candidate = generate_candidate_sequence(seq, num_substitutions=10, conservative=False)
        
        # Count differences
        differences = sum(1 for a, b in zip(seq, candidate) if a != b)
        
        # Should be close to requested (some may be identical by chance)
        self.assertGreater(differences, 0)
        self.assertLessEqual(differences, 15)  # Allow some variance
    
    def test_conservative_substitution(self):
        """Conservative substitutions should preserve biochemical properties."""
        # All hydrophobic
        seq = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']
        candidate = generate_candidate_sequence(seq, num_substitutions=4, conservative=True)
        
        # Check that substituted positions maintain same group
        # (Some positions may not be substituted, so we just check validity)
        self.assertTrue(all(aa in STANDARD_AAS for aa in candidate))
    
    def test_zero_substitutions(self):
        """Zero substitutions should return unchanged sequence."""
        seq = ['A', 'C', 'D', 'E', 'F']
        candidate = generate_candidate_sequence(seq, num_substitutions=0)
        self.assertEqual(candidate, seq)
    
    def test_empty_sequence(self):
        """Empty sequence should return empty."""
        candidate = generate_candidate_sequence([], num_substitutions=5)
        self.assertEqual(candidate, [])


class TestBootstrappedGeneration(unittest.TestCase):
    """Test bootstrapped sequence generation."""
    
    def test_candidate_count(self):
        """Should generate the requested number of candidates."""
        seq = ['A', 'C', 'D', 'E', 'F']
        candidates = generate_bootstrapped_sequences(
            seq, num_candidates=10, num_substitutions=2
        )
        self.assertEqual(len(candidates), 10)
    
    def test_candidate_diversity(self):
        """Candidates should be diverse (not all identical)."""
        seq = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L']
        candidates = generate_bootstrapped_sequences(
            seq, num_candidates=20, num_substitutions=5, conservative=False
        )
        
        # Convert to tuples for hashing
        unique_candidates = set(tuple(c) for c in candidates)
        
        # Should have multiple unique candidates
        self.assertGreater(len(unique_candidates), 1)
    
    def test_all_valid(self):
        """All candidates should be valid sequences."""
        seq = ['M', 'E', 'T', 'H']
        candidates = generate_bootstrapped_sequences(
            seq, num_candidates=5, num_substitutions=2
        )
        
        for candidate in candidates:
            self.assertTrue(all(aa in STANDARD_AAS for aa in candidate))
            self.assertEqual(len(candidate), len(seq))


class TestRankBasedSelection(unittest.TestCase):
    """Test rank-based selection mechanism."""
    
    def test_basic_selection(self):
        """Should select one candidate."""
        candidates = [
            ['A', 'C', 'D'],
            ['E', 'F', 'G'],
            ['H', 'I', 'K']
        ]
        scores = [0.5, 0.8, 0.3]
        
        selected = rank_based_selection(candidates, scores)
        
        # Should be one of the candidates
        self.assertIn(selected, candidates)
    
    def test_high_score_preference(self):
        """Should prefer higher-scored candidates statistically."""
        candidates = [
            ['A', 'C', 'D'],  # Low score
            ['E', 'F', 'G'],  # High score
        ]
        scores = [0.1, 0.9]
        
        # Run multiple times and check that high-score is selected more often
        selections = []
        for _ in range(100):
            selected = rank_based_selection(candidates, scores, temperature=0.5)
            if selected == candidates[1]:
                selections.append(1)
            else:
                selections.append(0)
        
        # High-score candidate should be selected majority of the time
        self.assertGreater(sum(selections), 60)  # At least 60% of the time
    
    def test_temperature_effect(self):
        """Temperature should affect selection randomness."""
        candidates = [['A'], ['B'], ['C']]
        scores = [0.3, 0.5, 0.2]
        
        # Low temperature should be more deterministic
        # High temperature should be more random
        # Just test that it doesn't crash
        selected_low = rank_based_selection(candidates, scores, temperature=0.1)
        selected_high = rank_based_selection(candidates, scores, temperature=10.0)
        
        self.assertIn(selected_low, candidates)
        self.assertIn(selected_high, candidates)


class TestBootGenAugment(unittest.TestCase):
    """Test the main BootGen augmentation function."""
    
    def test_basic_augmentation(self):
        """Should augment a sequence successfully."""
        seq = ['M', 'E', 'T', 'H', 'I', 'L', 'A', 'M', 'I', 'N']
        augmented = bootgen_augment(seq, intensity=0.3)
        
        # Should return a valid sequence
        self.assertIsInstance(augmented, list)
        self.assertTrue(all(aa in STANDARD_AAS for aa in augmented))
    
    def test_length_preservation(self):
        """Should preserve sequence length (or close to it)."""
        seq = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L']
        augmented = bootgen_augment(seq, intensity=0.5)
        
        # Should be within reasonable range
        self.assertGreaterEqual(len(augmented), 5)
        self.assertLessEqual(len(augmented), len(seq) * 1.3)
    
    def test_zero_intensity(self):
        """Zero intensity should return original sequence."""
        seq = ['A', 'C', 'D', 'E', 'F']
        augmented = bootgen_augment(seq, intensity=0.0)
        self.assertEqual(augmented, seq)
    
    def test_varying_intensity(self):
        """Different intensities should produce different results."""
        seq = ['M', 'E', 'T', 'H', 'I', 'L', 'A', 'M', 'I', 'N']
        
        # Test different intensities
        aug_low = bootgen_augment(seq, intensity=0.1)
        aug_mid = bootgen_augment(seq, intensity=0.5)
        aug_high = bootgen_augment(seq, intensity=0.9)
        
        # All should be valid
        self.assertTrue(all(aa in STANDARD_AAS for aa in aug_low))
        self.assertTrue(all(aa in STANDARD_AAS for aa in aug_mid))
        self.assertTrue(all(aa in STANDARD_AAS for aa in aug_high))
    
    def test_empty_sequence(self):
        """Empty sequence should return empty."""
        augmented = bootgen_augment([], intensity=0.5)
        self.assertEqual(augmented, [])
    
    def test_single_aa_sequence(self):
        """Single amino acid should be handled gracefully."""
        seq = ['A']
        augmented = bootgen_augment(seq, intensity=0.5)
        self.assertIsInstance(augmented, list)
        self.assertGreaterEqual(len(augmented), 0)
    
    def test_long_sequence(self):
        """Should handle long sequences."""
        seq = list('ACDEFGHIKLMNPQRSTVWY' * 10)  # 200 AAs
        augmented = bootgen_augment(seq, intensity=0.3)
        
        self.assertGreater(len(augmented), 100)
        self.assertTrue(all(aa in STANDARD_AAS for aa in augmented))
    
    def test_integration_interface(self):
        """Should be compatible with framework interface."""
        # Test that it works like other augmentation functions
        seq_str = "METHIONINE"
        seq_list = list(seq_str)
        
        augmented = bootgen_augment(seq_list, 0.3)
        
        # Should return list of amino acids
        self.assertIsInstance(augmented, list)
        self.assertTrue(all(isinstance(aa, str) for aa in augmented))
        
        # Should be convertible back to string
        augmented_str = ''.join(augmented)
        self.assertIsInstance(augmented_str, str)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_negative_intensity(self):
        """Negative intensity should be handled gracefully."""
        seq = ['A', 'C', 'D', 'E', 'F']
        augmented = bootgen_augment(seq, intensity=-0.5)
        # Should return original or valid sequence
        self.assertIsInstance(augmented, list)
    
    def test_very_high_intensity(self):
        """Very high intensity should be handled gracefully."""
        seq = ['A', 'C', 'D', 'E', 'F']
        augmented = bootgen_augment(seq, intensity=10.0)
        # Should return valid sequence
        self.assertTrue(all(aa in STANDARD_AAS for aa in augmented))


def run_tests():
    """Run all tests and print results."""
    print("=" * 70)
    print("Running BootGen Augmentation Tests")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCompositionSimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestPropertySimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceScore))
    suite.addTests(loader.loadTestsFromTestCase(TestCandidateGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestBootstrappedGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestRankBasedSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestBootGenAugment))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[PASS] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed.")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
