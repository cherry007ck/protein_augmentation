"""
Simple demonstration of PreIS augmentation.

This script shows PreIS in action with visual examples of:
1. Global segment swapping
2. Local token shuffling
3. Combined PreIS augmentation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentations.preis_augmentation import (
    preis_global_mixing,
    preis_local_mixing,
    preis_augment
)
from collections import Counter


def demo_global_mixing():
    """Demonstrate global segment swapping."""
    print("=" * 70)
    print("DEMO 1: Global Segment Swapping")
    print("=" * 70)
    
    # Use a sequence with clear patterns
    seq = list("AAAAA" + "BBBBB" + "CCCCC" + "DDDDD")
    print(f"\nOriginal sequence: {''.join(seq)}")
    print(f"Length: {len(seq)}, Pattern: 4 blocks of 5")
    
    # Apply global mixing a few times
    print("\nApplying global segment swapping (GAMMA_G=0.4):")
    for i in range(5):
        mixed = preis_global_mixing(seq.copy(), 0.4)
        print(f"  Trial {i+1}: {''.join(mixed)}")
    
    print("\nNote: Segments are swapped but composition preserved")


def demo_local_mixing():
    """Demonstrate local token shuffling."""
    print("\n" + "=" * 70)
    print("DEMO 2: Local Token Shuffling")
    print("=" * 70)
    
    # Use a sequence with unique tokens
    seq = list("ABCDEFGHIJKLMNOP")
    print(f"\nOriginal sequence: {''.join(seq)}")
    print(f"Length: {len(seq)}")
    
    # Apply local mixing a few times
    print("\nApplying local token shuffling (GAMMA_L=0.3):")
    for i in range(5):
        mixed = preis_local_mixing(seq.copy(), 0.3)
        print(f"  Trial {i+1}: {''.join(mixed)}")
    
    print("\nNote: Individual tokens are shuffled at random positions")


def demo_preis_augment():
    """Demonstrate full PreIS augmentation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Full PreIS Augmentation (Global + Local)")
    print("=" * 70)
    
    # Use a realistic protein sequence
    seq = list("METHYLAMINEKFPSTWYVARGLDEFGHILC")
    print(f"\nOriginal sequence: {''.join(seq)}")
    print(f"Length: {len(seq)}")
    print(f"Composition: {dict(Counter(seq))}")
    
    # Apply PreIS with different intensities
    print("\nApplying PreIS augmentation:")
    
    for intensity in [0.1, 0.5, 1.0, 2.0]:
        augmented = preis_augment(seq.copy(), intensity)
        print(f"\n  Intensity {intensity}:")
        print(f"    Original:  {''.join(seq)}")
        print(f"    Augmented: {''.join(augmented)}")
        print(f"    Composition preserved: {Counter(seq) == Counter(augmented)}")


def demo_framework_integration():
    """Demonstrate that PreIS works within the framework."""
    print("\n" + "=" * 70)
    print("DEMO 4: Framework Integration")
    print("=" * 70)
    
    try:
        # Import augmentation functions
        from augmentations.preis_augmentation import preis_augment
        
        print("\n[OK] Successfully imported preis_augment from augmentations")
        
        # Test it works like other augmentations
        seq = list("ARGLKFPSTWYV")
        result = preis_augment(seq, 0.5)
        
        print(f"\nTest sequence: {''.join(seq)}")
        print(f"Augmented:     {''.join(result)}")
        print(f"Length preserved: {len(seq) == len(result)}")
        print(f"Composition preserved: {Counter(seq) == Counter(result)}")
        
        # Check it's in the augmentation list
        with open("c:\\Users\\User\\Documents\\Masters_Project\\protein_augmentation\\example.py", 'r') as f:
            if 'preis_augment' in f.read():
                print("\n[OK] PreIS is registered in AUGMENTATION_FUNCTIONS")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")


def main():
    """Run all demos."""
    print("\n")
    print("#" * 70)
    print("# PreIS Augmentation - Interactive Demo")
    print("#" * 70)
    print("\nThis demo shows how PreIS performs simplified self-mixing")
    print("through global segment swapping and local token shuffling.\n")
    
    demo_global_mixing()
    demo_local_mixing()
    demo_preis_augment()
    demo_framework_integration()
    
    print("\n" + "#" * 70)
    print("# Demo Complete!")
    print("#" * 70)
    print("\nKey Takeaways:")
    print("  1. PreIS preserves sequence length and composition")
    print("  2. Creates diversity through structured permutation")
    print("  3. Intensity parameter controls augmentation strength")
    print("  4. Fully integrated with the augmentation framework")
    print("\nNext steps:")
    print("  - Test with actual protein datasets")
    print("  - Benchmark model performance with PreIS")
    print("  - Combine with other augmentations for best results")
    print()


if __name__ == "__main__":
    main()
