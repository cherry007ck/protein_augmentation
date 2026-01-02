"""
Demonstration of BootGen Augmentation

This script demonstrates the BootGen-inspired augmentation technique
and its key features:
- Bootstrapped candidate generation
- Proxy-based quality scoring
- Rank-based selection

Reference: "Bootstrapped Training of Score-Conditioned Generator for 
Offline Design of Biological Sequences" (NeurIPS 2023)
"""

import random
import numpy as np
from typing import List
from collections import Counter

# Import BootGen functions
from augmentations.bootgen import (
    bootgen_augment,
    compute_sequence_score,
    generate_bootstrapped_sequences,
    compute_composition_similarity,
    compute_property_similarity,
    AMINO_ACID_GROUPS,
    AA_TO_GROUP
)


def print_header(title: str, width: int = 70):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_sequence_comparison(original: List[str], augmented: List[str]):
    """Print a visual comparison of original and augmented sequences."""
    print(f"\nOriginal:  {''.join(original)}")
    print(f"Augmented: {''.join(augmented)}")
    
    # Highlight differences
    differences = []
    for i, (o, a) in enumerate(zip(original, augmented)):
        if o != a:
            differences.append((i, o, a))
    
    if differences:
        print(f"\nDifferences: {len(differences)} positions")
        for pos, orig, aug in differences[:10]:  # Show first 10
            orig_group = AA_TO_GROUP.get(orig, 'unknown')
            aug_group = AA_TO_GROUP.get(aug, 'unknown')
            conserved = "[Y]" if orig_group == aug_group else "[N]"
            print(f"  Position {pos}: {orig} -> {aug} (Group: {orig_group} -> {aug_group}) {conserved}")
        if len(differences) > 10:
            print(f"  ... and {len(differences) - 10} more")
    else:
        print("\nNo differences (sequences are identical)")


def print_composition(sequence: List[str]):
    """Print amino acid composition statistics."""
    counts = Counter(sequence)
    total = len(sequence)
    
    print("\nAmino Acid Composition:")
    # Group by biochemical properties
    for group_name, group_aas in AMINO_ACID_GROUPS.items():
        group_count = sum(counts.get(aa, 0) for aa in group_aas)
        percentage = (group_count / total * 100) if total > 0 else 0
        print(f"  {group_name:12s}: {group_count:3d} ({percentage:5.1f}%)")


def demo_basic_usage():
    """Demonstrate basic BootGen augmentation usage."""
    print_header("Demo 1: Basic BootGen Augmentation")
    
    # Example protein sequence
    sequence_str = "METHIONYLGLUTAMINYLARGINYLTYROSYLGLUTAMYL"
    sequence = list(sequence_str)
    
    print(f"\nOriginal sequence: {sequence_str}")
    print(f"Length: {len(sequence)} amino acids")
    
    # Apply augmentation with moderate intensity
    intensity = 0.3
    print(f"\nApplying BootGen augmentation (intensity={intensity})...")
    augmented = bootgen_augment(sequence, intensity)
    
    print_sequence_comparison(sequence, augmented)
    
    # Compute score
    score = compute_sequence_score(sequence, augmented)
    print(f"\nQuality Score: {score:.3f}")


def demo_intensity_comparison():
    """Demonstrate effect of different intensity levels."""
    print_header("Demo 2: Intensity Comparison")
    
    sequence_str = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    sequence = list(sequence_str)
    
    print(f"Original sequence: {sequence_str}")
    
    intensities = [0.1, 0.3, 0.5, 0.8]
    
    for intensity in intensities:
        print(f"\n{'-' * 70}")
        print(f"Intensity: {intensity}")
        
        augmented = bootgen_augment(sequence, intensity)
        
        # Count differences
        differences = sum(1 for a, b in zip(sequence, augmented) if a != b)
        diff_percent = (differences / len(sequence) * 100) if len(sequence) > 0 else 0
        
        # Compute scores
        score = compute_sequence_score(sequence, augmented)
        comp_sim = compute_composition_similarity(sequence, augmented)
        prop_sim = compute_property_similarity(sequence, augmented)
        
        print(f"  Differences: {differences}/{len(sequence)} ({diff_percent:.1f}%)")
        print(f"  Overall Score: {score:.3f}")
        print(f"  Composition Similarity: {comp_sim:.3f}")
        print(f"  Property Similarity: {prop_sim:.3f}")


def demo_bootstrapped_generation():
    """Demonstrate bootstrapped candidate generation."""
    print_header("Demo 3: Bootstrapped Candidate Generation")
    
    sequence = list("ACDEFGHIKLMNPQRST")
    print(f"Original sequence: {''.join(sequence)}")
    
    # Generate multiple candidates
    num_candidates = 10
    num_substitutions = 5
    
    print(f"\nGenerating {num_candidates} candidates with {num_substitutions} substitutions each...")
    candidates = generate_bootstrapped_sequences(
        sequence, 
        num_candidates=num_candidates,
        num_substitutions=num_substitutions,
        conservative=True
    )
    
    # Score each candidate
    print("\nCandidate Scores:")
    scores = []
    for i, candidate in enumerate(candidates):
        score = compute_sequence_score(sequence, candidate)
        scores.append(score)
        
        # Count differences
        diffs = sum(1 for a, b in zip(sequence, candidate) if a != b)
        
        print(f"  Candidate {i+1}: Score={score:.3f}, Differences={diffs}, Seq={''.join(candidate)}")
    
    # Show statistics
    print(f"\nScore Statistics:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std:  {np.std(scores):.3f}")
    print(f"  Min:  {np.min(scores):.3f}")
    print(f"  Max:  {np.max(scores):.3f}")


def demo_property_preservation():
    """Demonstrate biochemical property preservation."""
    print_header("Demo 4: Biochemical Property Preservation")
    
    # Create a sequence rich in hydrophobic amino acids
    sequence = list("AAAVVVIIILLLMMMFFFWWWPPP")
    
    print(f"Original sequence: {''.join(sequence)}")
    print_composition(sequence)
    
    # Augment with moderate intensity
    augmented = bootgen_augment(sequence, intensity=0.4)
    
    print(f"\nAugmented sequence: {''.join(augmented)}")
    print_composition(augmented)
    
    # Compare property similarity
    prop_sim = compute_property_similarity(sequence, augmented)
    print(f"\nProperty Similarity: {prop_sim:.3f}")
    print("(Higher values indicate better preservation of biochemical properties)")


def demo_multiple_runs():
    """Demonstrate stochasticity in BootGen augmentation."""
    print_header("Demo 5: Multiple Augmentation Runs")
    
    sequence = list("METHIONINE")
    print(f"Original sequence: {''.join(sequence)}")
    
    num_runs = 5
    intensity = 0.4
    
    print(f"\nRunning BootGen {num_runs} times (intensity={intensity}):")
    
    all_augmented = []
    for i in range(num_runs):
        augmented = bootgen_augment(sequence, intensity)
        all_augmented.append(augmented)
        
        score = compute_sequence_score(sequence, augmented)
        diffs = sum(1 for a, b in zip(sequence, augmented) if a != b)
        
        print(f"\nRun {i+1}:")
        print(f"  Sequence: {''.join(augmented)}")
        print(f"  Score: {score:.3f}")
        print(f"  Differences: {diffs}")
    
    # Check diversity
    unique_sequences = set(''.join(aug) for aug in all_augmented)
    print(f"\nDiversity: {len(unique_sequences)}/{num_runs} unique sequences")


def demo_framework_integration():
    """Demonstrate integration with the augmentation framework."""
    print_header("Demo 6: Framework Integration")
    
    print("BootGen is now integrated into the augmentation framework!")
    print("\nIt can be used alongside other augmentation techniques:")
    
    # Import from example.py would normally work
    # from example import AUGMENTATION_FUNCTIONS
    # For demo purposes, we'll simulate
    
    sequence = list("PROTEINSEQUENCE")
    print(f"\nOriginal sequence: {''.join(sequence)}")
    
    # Show BootGen usage
    print("\nUsing BootGen:")
    aug_bootgen = bootgen_augment(sequence, intensity=0.3)
    print(f"  Result: {''.join(aug_bootgen)}")
    
    print("\n[OK] BootGen can be used in:")
    print("  - Random augmentation selection (example.py)")
    print("  - Policy-based augmentation (APA framework)")
    print("  - Custom training pipelines")
    print("  - Data preprocessing workflows")


def main():
    """Run all demonstrations."""
    print_header("BootGen Augmentation Demonstration", width=70)
    print("\nBootGen: Bootstrapped Training of Score-Conditioned Generator")
    print("Reference: NeurIPS 2023")
    print("Implementation: Simplified for protein sequence augmentation")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run all demos
    demos = [
        demo_basic_usage,
        demo_intensity_comparison,
        demo_bootstrapped_generation,
        demo_property_preservation,
        demo_multiple_runs,
        demo_framework_integration
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n[ERROR] Error in demo: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print_header("Summary", width=70)
    print("\n[OK] BootGen augmentation demonstrated successfully!")
    print("\nKey Features:")
    print("  - Bootstrapped candidate generation")
    print("  - Quality-aware selection via proxy scoring")
    print("  - Rank-based probabilistic sampling")
    print("  - Biochemical property preservation")
    print("  - Configurable intensity control")
    print("\nUsage in framework:")
    print("  from augmentations.bootgen import bootgen_augment")
    print("  augmented = bootgen_augment(sequence, intensity=0.3)")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
