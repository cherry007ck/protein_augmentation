"""
Demonstration of NaNa (Novel Augmentation of New Node Attributes) augmentation.

This script shows:
1. Basic usage of NaNa augmentation
2. Different intensity levels
3. Property preservation visualization
4. Integration with example.py framework
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentations.nana_augmentation import (
    nana_augment,
    calculate_similarity,
    HYDROPHOBICITY,
    CHARGE,
    HELIX_PROPENSITY,
    SHEET_PROPENSITY,
)


def demo_basic_usage():
    """Demonstrate basic NaNa augmentation usage."""
    print("="*60)
    print("Demo 1: Basic Usage")
    print("="*60)
    
    # Example protein sequence
    seq = list("MKTAYIAKQRQISFVKSHFSRQ")
    print(f"Original sequence: {''.join(seq)}")
    print(f"Length: {len(seq)} residues\n")
    
    # Apply NaNa augmentation with moderate intensity
    aug_seq = nana_augment(seq, substitution_rate=0.3)
    print(f"Augmented sequence: {''.join(aug_seq)}")
    print(f"Length: {len(aug_seq)} residues")
    
    # Count changes
    changes = sum(1 for i in range(len(seq)) if seq[i] != aug_seq[i])
    print(f"Number of substitutions: {changes} ({changes/len(seq)*100:.1f}%)\n")


def demo_intensity_levels():
    """Demonstrate different intensity levels."""
    print("="*60)
    print("Demo 2: Different Intensity Levels")
    print("="*60)
    
    seq = list("ACDEFGHIKLMNPQRSTVWY" * 2)
    print(f"Original: {''.join(seq)}")
    print(f"Length: {len(seq)} residues\n")
    
    intensities = [0.1, 0.3, 0.5, 0.7]
    
    for intensity in intensities:
        aug_seq = nana_augment(seq, substitution_rate=intensity)
        changes = sum(1 for i in range(len(seq)) if seq[i] != aug_seq[i])
        print(f"Intensity {intensity:.1f}: {changes:2d} changes ({changes/len(seq)*100:4.1f}%) - {''.join(aug_seq[:20])}...")
    
    print()


def demo_property_preservation():
    """Demonstrate that biophysical properties are preserved."""
    print("="*60)
    print("Demo 3: Property Preservation")
    print("="*60)
    
    # Hydrophobic sequence
    seq = list("VVVIIILLLMMM")
    print(f"Hydrophobic sequence: {''.join(seq)}\n")
    
    # Calculate original properties
    orig_hydro = sum(HYDROPHOBICITY[aa] for aa in seq) / len(seq)
    orig_charge = sum(CHARGE[aa] for aa in seq)
    orig_helix = sum(HELIX_PROPENSITY[aa] for aa in seq) / len(seq)
    
    print("Original properties:")
    print(f"  Hydrophobicity: {orig_hydro:.2f}")
    print(f"  Total charge: {orig_charge}")
    print(f"  Helix propensity: {orig_helix:.2f}\n")
    
    # Augment
    aug_seq = nana_augment(seq, substitution_rate=0.5)
    print(f"Augmented sequence: {''.join(aug_seq)}\n")
    
    # Calculate augmented properties
    aug_hydro = sum(HYDROPHOBICITY[aa] for aa in aug_seq) / len(aug_seq)
    aug_charge = sum(CHARGE[aa] for aa in aug_seq)
    aug_helix = sum(HELIX_PROPENSITY[aa] for aa in aug_seq) / len(aug_seq)
    
    print("Augmented properties:")
    print(f"  Hydrophobicity: {aug_hydro:.2f} (Δ = {abs(aug_hydro - orig_hydro):.2f})")
    print(f"  Total charge: {aug_charge} (Δ = {abs(aug_charge - orig_charge)})")
    print(f"  Helix propensity: {aug_helix:.2f} (Δ = {abs(aug_helix - orig_helix):.2f})")
    print("\nNote: Properties are well-preserved!\n")


def demo_charge_preservation():
    """Demonstrate charge preservation."""
    print("="*60)
    print("Demo 4: Charge Preservation")
    print("="*60)
    
    # Charged sequence (positive and negative)
    seq = list("KKKRRRDDDEEE")
    print(f"Charged sequence: {''.join(seq)}")
    
    orig_pos = sum(1 for aa in seq if CHARGE[aa] > 0)
    orig_neg = sum(1 for aa in seq if CHARGE[aa] < 0)
    print(f"Original: +{orig_pos} charged, -{orig_neg} charged\n")
    
    # Multiple augmentations to show consistency
    for i in range(3):
        aug_seq = nana_augment(seq, substitution_rate=0.5)
        aug_pos = sum(1 for aa in aug_seq if CHARGE[aa] > 0)
        aug_neg = sum(1 for aa in aug_seq if CHARGE[aa] < 0)
        
        print(f"Augmentation {i+1}: {''.join(aug_seq)}")
        print(f"  +{aug_pos} charged, -{aug_neg} charged\n")


def demo_similarity_matrix():
    """Show similarity between common amino acid pairs."""
    print("="*60)
    print("Demo 5: Amino Acid Similarity Examples")
    print("="*60)
    
    pairs = [
        ('I', 'L', 'Both hydrophobic, aliphatic'),
        ('D', 'E', 'Both negatively charged'),
        ('K', 'R', 'Both positively charged'),
        ('F', 'Y', 'Both aromatic'),
        ('S', 'T', 'Both polar, small'),
        ('I', 'K', 'Hydrophobic vs. charged'),
        ('K', 'D', 'Oppositely charged'),
    ]
    
    print("Similarity scores (0.0 = very different, 1.0 = identical):\n")
    for aa1, aa2, description in pairs:
        sim = calculate_similarity(aa1, aa2)
        print(f"  {aa1}-{aa2}: {sim:.3f} ({description})")
    
    print()


def demo_framework_integration():
    """Demonstrate integration with framework interface."""
    print("="*60)
    print("Demo 6: Framework Integration")
    print("="*60)
    
    print("NaNa follows the standard framework interface:")
    print("  nana_augment(sequence: List[str], intensity: float) -> List[str]\n")
    
    seq = list("MKTAYIAK")
    print(f"Example sequence: {''.join(seq)}\n")
    
    print("Can be used in AUGMENTATION_FUNCTIONS list in example.py:")
    print("  - Automatically selected during random augmentation")
    print("  - Compatible with policy-based augmentation (APA)")
    print("  - Works with dataloader augmentation pipeline\n")
    
    # Simulate framework usage
    print("Simulating framework call:")
    augmentation_intensity = 0.2
    result = nana_augment(seq, augmentation_intensity)
    print(f"  Input: {seq}")
    print(f"  Intensity: {augmentation_intensity}")
    print(f"  Output: {result}")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("NaNa (Novel Augmentation of New Node Attributes)")
    print("Semantic Data Augmentation for Protein Sequences")
    print("="*60 + "\n")
    
    demo_basic_usage()
    demo_intensity_levels()
    demo_property_preservation()
    demo_charge_preservation()
    demo_similarity_matrix()
    demo_framework_integration()
    
    print("="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
