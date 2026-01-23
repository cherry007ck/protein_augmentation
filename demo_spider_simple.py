"""
Simple demonstration of Spider augmentation functionality.

This standalone script demonstrates the core Spider features without
requiring the full framework dependencies.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

# Set random seed for reproducibility
random.seed(42)

from augmentations.spider_augmentation import (
    spider_augment,
    random_substitute_residues,
    random_insert_residues
)


def main():
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "  Spider Augmentation Demonstration".center(68) + "|")
    print("|" + "  Reference: Spider Neurotoxic Peptide Paper".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")

    
    # Demo 1: Basic Spider Functionality
    print("\n" + "=" * 70)
    print("1. Basic Spider Functionality")
    print("=" * 70)
    
    protein_seq = list("METHYLAMINE")
    print(f"\nOriginal protein: {''.join(protein_seq)}")
    print(f"Length: {len(protein_seq)} amino acids")
    
    # Show Spider with different intensities
    print("\nApplying Spider with different intensities:")
    for intensity in [0.0, 0.2, 0.5, 0.8]:
        augmented = spider_augment(protein_seq, intensity)
        length_change = len(augmented) - len(protein_seq)
        print(f"  Intensity {intensity:.1f}: {''.join(augmented):25s} (Delta length: {length_change:+d})")
    
    # Demo 2: Substitution vs Insertion
    print("\n" + "=" * 70)
    print("2. Substitution vs Insertion Components")
    print("=" * 70)
    
    test_seq = list("GAGAGA")
    print(f"\nOriginal sequence: {''.join(test_seq)} (length: {len(test_seq)})")
    
    print("\nSubstitution only (rate=0.5):")
    for i in range(3):
        result = random_substitute_residues(test_seq, 0.5)
        num_changes = sum(1 for a, b in zip(test_seq, result) if a != b)
        print(f"  {i+1}. {''.join(result):20s} ({num_changes}/{len(test_seq)} changed)")
    
    print("\nInsertion only (rate=0.5):")
    for i in range(3):
        result = random_insert_residues(test_seq, 0.5)
        num_added = len(result) - len(test_seq)
        print(f"  {i+1}. {''.join(result):20s} (+{num_added} residues)")
    
    # Demo 3: Combined Effect
    print("\n" + "=" * 70)
    print("3. Combined Substitution + Insertion")
    print("=" * 70)
    
    original = list("ARGLKM")
    print(f"\nOriginal protein: {''.join(original)} (length: {len(original)})")
    
    print("\nStep-by-step Spider process (intensity=0.6):")
    print(f"  1. Original:       {''.join(original)}")
    
    # Apply with intensity 0.6
    # This means substitution_rate=0.3, insertion_rate=0.3
    result_sub = random_substitute_residues(original, 0.3)
    num_subs = sum(1 for a, b in zip(original, result_sub) if a != b)
    print(f"  2. Substitute:     {''.join(result_sub)} ({num_subs} changed)")
    
    result_final = random_insert_residues(result_sub, 0.3)
    num_inserts = len(result_final) - len(result_sub)
    print(f"  3. Insert:         {''.join(result_final)} (+{num_inserts} inserted)")
    
    print(f"\n  Final length: {len(result_final)} (original: {len(original)})")
    
    # Demo 4: Multiple Augmentations
    print("\n" + "=" * 70)
    print("4. Multiple Augmentations of Same Sequence")
    print("=" * 70)
    
    base_seq = list("METHYLKFPSTWYV")
    print(f"\nOriginal: {''.join(base_seq)}")
    print("\nGenerating 5 different augmentations (intensity=0.4):")
    
    for i in range(5):
        augmented = spider_augment(base_seq, 0.4)
        num_changes = sum(1 for a, b in zip(base_seq, augmented[:len(base_seq)]) if a != b)
        length_diff = len(augmented) - len(base_seq)
        print(f"  {i+1}. {''.join(augmented):30s} ({num_changes} subs, {length_diff:+d} length)")
    
    # Demo 5: Intensity Comparison
    print("\n" + "=" * 70)
    print("5. Effect of Different Intensities")
    print("=" * 70)
    
    sequence = list("ACDEFGHIKLMNPQRSTVWY")
    print(f"\nOriginal (all 20 AAs): {''.join(sequence)}")
    print(f"Length: {len(sequence)}\n")
    
    intensities = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("Intensity | Length | Changes | Result")
    print("-" * 70)
    
    for intensity in intensities:
        result = spider_augment(sequence, intensity)
        changes = sum(1 for a, b in zip(sequence, result[:len(sequence)]) if a != b)
        print(f"   {intensity:.1f}    |   {len(result):2d}   |   {changes:2d}    | {''.join(result[:30])}")
    
    # Demo 6: Key Advantages
    print("\n" + "=" * 70)
    print("6. Key Features of Spider Augmentation")
    print("=" * 70)
    
    print("""
Spider augmentation combines two powerful operations:

  1. Substitution: Creates point mutations (single AA changes)
     -> Mimics natural variation and mutations
     -> Can test functional importance of specific residues
  
  2. Insertion: Adds new residues to the sequence  
     -> Increases sequence diversity significantly
     -> Can explore longer sequence variants
     -> Useful for discovering novel functional peptides

Combined Effect:
  [OK] High sequence diversity
  [OK] Variable-length outputs (more realistic for some tasks)
  [OK] Simple and fast (no complex dependencies)
  [OK] Biology-inspired (validated for neurotoxic peptide prediction)
    
Framework Integration:
  [OK] Compatible with existing augmentation interface
  [OK] Works with policy-based augmentation strategies
  [OK] Can be combined with other augmentation techniques
    """)
    
    # Demo 7: Comparison with Original
    print("\n" + "=" * 70)
    print("7. Comparison: Original vs Augmented")
    print("=" * 70)
    
    original_peptide = list("MGLFFIASTLAVA")
    print(f"\nOriginal peptide:  {''.join(original_peptide)}")
    
    augmented_peptide = spider_augment(original_peptide, 0.3)
    print(f"Augmented peptide: {''.join(augmented_peptide)}")
    
    # Show detailed comparison
    print("\nDetailed comparison:")
    print("  Position-by-position (first {} positions):".format(min(len(original_peptide), len(augmented_peptide))))
    
    for i, (orig, aug) in enumerate(zip(original_peptide, augmented_peptide)):
        status = "[OK]" if orig == aug else "[X] changed"
        print(f"    Position {i+1:2d}: {orig} -> {aug}  {status}")
    
    if len(augmented_peptide) > len(original_peptide):
        print(f"\n  New insertions at end:")
        for i in range(len(original_peptide), len(augmented_peptide)):
            print(f"    Position {i+1}: {augmented_peptide[i]} (inserted)")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Spider Augmentation creates sequence diversity by:
  1. Randomly substituting amino acids with other amino acids
  2. Randomly inserting new amino acids at random positions

Key parameters:
  • intensity: Controls both substitution and insertion rates (0.0-1.0)
    - substitution_rate = intensity * 0.5
    - insertion_rate = intensity * 0.5

Benefits:
  [OK] Increases training data diversity
  [OK] Can discover novel functional variants
  [OK] Simple and fast implementation
  [OK] No external dependencies (unlike BLAST filtering)
  
Integration:
  [OK] Successfully integrated into the augmentation framework
  [OK] Compatible with existing augmentation interface
  [OK] Can be used in automated augmentation policy search
""")
    
    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
