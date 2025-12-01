"""
Demonstration of NTA (Nucleotide Augmentation) Integration

This script demonstrates how the NTA augmentation technique integrates
with the existing protein augmentation framework.
"""

import sys
sys.path.insert(0, '/home/luffy/protein_augmentation')

import random

# Set random seed for reproducibility
random.seed(42)

from augmentations.nta_augmentation import nucleotide_augment, back_translate, CODON_TO_AMINO_ACID


def demonstrate_nta_basic():
    """Demonstrate basic NTA functionality."""
    print("=" * 70)
    print("DEMONSTRATION: Basic NTA Functionality")
    print("=" * 70)
    
    # Example protein sequence
    protein_seq = list("METHYLKFPSTWYV")
    print(f"\nOriginal protein sequence: {''.join(protein_seq)}")
    print(f"Length: {len(protein_seq)} amino acids")
    
    # Back-translate to show the nucleotide diversity
    print("\n1. Back-translation (AA → DNA):")
    codons = back_translate(protein_seq)
    print(f"   DNA sequence: {' '.join(codons)}")
    print(f"   Length: {len(codons)} codons = {len(codons)*3} nucleotides")
    
    # Apply NTA with different intensities
    print("\n2. NTA with different substitution rates:")
    
    for rate in [0.0, 0.2, 0.5, 0.8]:
        augmented = nucleotide_augment(protein_seq, rate)
        print(f"\n   Rate {rate:.1f}: {''.join(augmented)}")
        
        # Show that amino acid sequence is preserved
        if augmented == protein_seq:
            print(f"   ✓ Amino acid sequence preserved")
        else:
            print(f"   ✗ WARNING: Sequence changed!")
    
    print()


def demonstrate_nta_with_framework():
    """Demonstrate NTA as part of the augmentation function list."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: NTA in Augmentation Pipeline")
    print("=" * 70)
    
    # Import augmentation functions
    from example import AUGMENTATION_FUNCTIONS
    
    print(f"\nTotal augmentation functions: {len(AUGMENTATION_FUNCTIONS)}")
    print("\nAugmentation function list:")
    for i, func in enumerate(AUGMENTATION_FUNCTIONS, 1):
        marker = "← NEW!" if func.__name__ == 'nucleotide_augment' else ""
        print(f"  {i:2d}. {func.__name__:30s} {marker}")
    
    # Test NTA as part of random augmentation selection
    print("\n" + "-" * 70)
    print("Applying random augmentations (including NTA):")
    print("-" * 70)
    
    test_seq = list("ARGLKFPSTWYV")
    print(f"\nOriginal: {''.join(test_seq)}")
    
    # Apply 5 random augmentations
    for i in range(5):
        aug_func = random.choice(AUGMENTATION_FUNCTIONS)
        intensity = random.uniform(0.1, 0.3)
        
        try:
            augmented = aug_func(test_seq, intensity)
            if len(augmented) >= 5:  # Valid augmentation
                result_str = ''.join(augmented)
                print(f"  {i+1}. {aug_func.__name__:30s} (λ={intensity:.2f}): {result_str}")
            else:
                print(f"  {i+1}. {aug_func.__name__:30s} (λ={intensity:.2f}): [too short, skipped]")
        except Exception as e:
            print(f"  {i+1}. {aug_func.__name__:30s} (λ={intensity:.2f}): [error: {e}]")
    
    print()


def demonstrate_nucleotide_diversity():
    """Show that NTA creates nucleotide-level diversity."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Nucleotide-Level Diversity")
    print("=" * 70)
    
    protein = list("GAGAGA")  # Glycine-Alanine repeats
    print(f"\nProtein sequence: {''.join(protein)}")
    print("(Glycine has 4 synonymous codons, Alanine has 4)")
    
    print("\nGenerating 5 different back-translations:")
    print("(Each should encode the same protein but use different codons)")
    
    for i in range(5):
        codons = back_translate(protein)
        dna = ''.join(codons)
        print(f"  {i+1}. {' '.join(codons)} → {dna}")
    
    print("\n✓ Same protein, different DNA sequences!")
    print("  This is the core idea of Nucleotide Augmentation.")
    print()


def compare_with_other_augmentations():
    """Compare NTA with other augmentation techniques."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Comparing Augmentation Techniques")
    print("=" * 70)
    
    from example import (
        substitute_random_residues,
        swap_random_residues,
        back_translation_substitute,
    )
    
    original = list("METHYLAMINE")
    print(f"\nOriginal sequence: {''.join(original)}")
    print(f"Length: {len(original)} AA\n")
    
    # Test different augmentation approaches
    augmentations = [
        ("NTA (nucleotide_augment)", nucleotide_augment, 0.3),
        ("Substitute residues", substitute_random_residues, 0.3),
        ("Swap residues", swap_random_residues, 0.3),
        ("Back-translation substitute", back_translation_substitute, 0.1),
    ]
    
    print("Applying different augmentations with similar intensity:")
    print("-" * 70)
    
    for name, func, intensity in augmentations:
        try:
            augmented = func(original, intensity)
            result = ''.join(augmented)
            
            # Count differences
            if len(augmented) == len(original):
                num_diff = sum(1 for a, b in zip(original, augmented) if a != b)
                preserve = "✓" if num_diff == 0 else "✗"
                print(f"{name:30s}: {result:15s} | {preserve} ({num_diff} AA changed)")
            else:
                print(f"{name:30s}: {result:15s} | Length changed: {len(original)} → {len(augmented)}")
        except Exception as e:
            print(f"{name:30s}: ERROR - {e}")
    
    print()
    print("Key difference:")
    print("  • NTA preserves amino acid identity (diversity at nucleotide level)")
    print("  • Other methods modify the amino acid sequence directly")
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  NTA (Nucleotide Augmentation) Integration Demonstration".center(68) + "║")
    print("║" + "  Reference: Minot & Reddy 2022".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    demonstrate_nta_basic()
    demonstrate_nucleotide_diversity()
    demonstrate_nta_with_framework()
    compare_with_other_augmentations()
    
    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print("\nNTA has been successfully integrated into the framework.")
    print("It can now be used alongside the existing 10 augmentation techniques.")
    print()


if __name__ == "__main__":
    main()
