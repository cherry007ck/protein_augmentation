"""
Demonstration of IMAEN (Interpretable Molecular Augmentation) for protein sequences.

This script demonstrates the property-aware amino acid substitution technique
and shows how it preserves biochemical properties while creating sequence diversity.
"""

from augmentations.imaen import (
    imaen_simple,
    imaen_augment,
    PROPERTY_GROUPS,
    HYDROPHOBIC_NONPOLAR,
    POLAR_UNCHARGED,
    POSITIVELY_CHARGED,
    NEGATIVELY_CHARGED,
    AROMATIC
)


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_sequence_comparison(original, augmented, title="Comparison"):
    """Print original and augmented sequences side by side."""
    print(f"\n{title}:")
    print(f"  Original:  {''.join(original)}")
    print(f"  Augmented: {''.join(augmented)}")
    
    # Calculate and display differences
    differences = sum(1 for a, b in zip(original, augmented) if a != b)
    diff_percentage = (differences / len(original)) * 100 if original else 0
    
    print(f"  Differences: {differences}/{len(original)} ({diff_percentage:.1f}%)")
    
    # Show position-by-position changes
    if differences > 0 and len(original) < 50:
        changes = []
        for i, (orig, aug) in enumerate(zip(original, augmented)):
            if orig != aug:
                changes.append(f"    Position {i}: {orig} -> {aug}")
        
        if changes:
            print("  Changes:")
            for change in changes[:10]:  # Limit to first 10 changes
                print(change)
            if len(changes) > 10:
                print(f"    ... and {len(changes) - 10} more")


def analyze_property_preservation(original, augmented):
    """Analyze how well biochemical properties are preserved."""
    print("\n  Property Analysis:")
    
    # Count property preservation
    same_property = 0
    property_changed = 0
    
    for orig, aug in zip(original, augmented):
        if orig == aug:
            same_property += 1
        elif aug in PROPERTY_GROUPS.get(orig, set()):
            same_property += 1  # Conservative substitution
        else:
            property_changed += 1
    
    total = len(original)
    preservation_rate = (same_property / total * 100) if total > 0 else 0
    
    print(f"    Property preserved: {same_property}/{total} ({preservation_rate:.1f}%)")
    print(f"    Property changed:   {property_changed}/{total} ({(100-preservation_rate):.1f}%)")


def demo_basic_augmentation():
    """Demonstrate basic IMAEN augmentation."""
    print_section_header("Demo 1: Basic IMAEN Augmentation")
    
    # Use a realistic protein sequence
    sequence = list("MKTAYIAKQRQISFVKSHFSRQLEERLGLEVCDLEIR")
    
    print(f"\nOriginal sequence length: {len(sequence)} amino acids")
    print(f"Original: {''.join(sequence)}")
    
    # Apply augmentation at different intensities
    intensities = [0.1, 0.3, 0.5]
    
    for intensity in intensities:
        augmented = imaen_simple(sequence, intensity=intensity)
        print_sequence_comparison(sequence, augmented, 
                                 f"\nIntensity {intensity}")
        analyze_property_preservation(sequence, augmented)


def demo_property_biases():
    """Demonstrate property-biased augmentation."""
    print_section_header("Demo 2: Property-Biased Augmentation")
    
    sequence = list("MKTAYIAKQRQISFVKSHFSRQLEERLGLEVCDLEIR")
    intensity = 0.4
    
    print(f"\nOriginal: {''.join(sequence)}")
    print(f"Testing different property biases at intensity {intensity}:\n")
    
    biases = [
        ('random', 'Random selection'),
        ('hydrophobic', 'Hydrophobic residues preferred'),
        ('polar', 'Polar residues preferred'),
        ('charged', 'Charged residues preferred'),
        ('aromatic', 'Aromatic residues preferred')
    ]
    
    for bias_name, description in biases:
        augmented = imaen_augment(sequence, intensity=intensity, 
                                 property_bias=bias_name)
        print(f"\n{description} (bias='{bias_name}'):")
        print(f"  Augmented: {''.join(augmented)}")
        
        differences = sum(1 for a, b in zip(sequence, augmented) if a != b)
        print(f"  Differences: {differences}/{len(sequence)}")


def demo_conservative_vs_diverse():
    """Demonstrate conservative vs. diverse substitution modes."""
    print_section_header("Demo 3: Conservative vs. Diverse Substitution")
    
    sequence = list("AVILMFWPKSTNQCYG")  # Mixed properties
    intensity = 0.5
    
    print(f"\nOriginal: {''.join(sequence)}")
    print("\nConservative mode (preserves biochemical properties):")
    
    conservative = imaen_augment(sequence, intensity=intensity, 
                                 conservative=True)
    print(f"  Augmented: {''.join(conservative)}")
    analyze_property_preservation(sequence, conservative)
    
    print("\nDiverse mode (allows broader substitutions):")
    diverse = imaen_augment(sequence, intensity=intensity, 
                           conservative=False)
    print(f"  Augmented: {''.join(diverse)}")
    analyze_property_preservation(sequence, diverse)


def demo_specific_properties():
    """Demonstrate augmentation of sequences with specific properties."""
    print_section_header("Demo 4: Property-Specific Sequences")
    
    test_sequences = [
        (list("AVILMFWP"), "Hydrophobic"),
        (list("STNQCYG"), "Polar uncharged"),
        (list("KRHDE"), "Charged"),
        (list("FYW"), "Aromatic")
    ]
    
    intensity = 0.5
    
    for seq, description in test_sequences:
        print(f"\n{description} sequence:")
        augmented = imaen_simple(seq, intensity=intensity)
        print_sequence_comparison(seq, augmented)
        
        # Check what substitutions were made
        for i, (orig, aug) in enumerate(zip(seq, augmented)):
            if orig != aug:
                similar = PROPERTY_GROUPS.get(orig, set())
                is_similar = aug in similar
                print(f"    {orig} -> {aug} {'(similar property)' if is_similar else '(different)'}")


def demo_realistic_protein():
    """Demonstrate augmentation on a realistic protein sequence."""
    print_section_header("Demo 5: Realistic Protein Sequence")
    
    # Insulin A chain (human)
    insulin_a = list("GIVEQCCTSICSLYQLENYCN")
    
    print("\nInsulin A chain (human):")
    print(f"Original: {''.join(insulin_a)}")
    print(f"Length: {len(insulin_a)} amino acids")
    
    # Show property composition
    hydrophobic = sum(1 for aa in insulin_a if aa in HYDROPHOBIC_NONPOLAR)
    polar = sum(1 for aa in insulin_a if aa in POLAR_UNCHARGED)
    positive = sum(1 for aa in insulin_a if aa in POSITIVELY_CHARGED)
    negative = sum(1 for aa in insulin_a if aa in NEGATIVELY_CHARGED)
    
    print(f"\nProperty composition:")
    print(f"  Hydrophobic: {hydrophobic} ({hydrophobic/len(insulin_a)*100:.1f}%)")
    print(f"  Polar:       {polar} ({polar/len(insulin_a)*100:.1f}%)")
    print(f"  Positive:    {positive} ({positive/len(insulin_a)*100:.1f}%)")
    print(f"  Negative:    {negative} ({negative/len(insulin_a)*100:.1f}%)")
    
    # Generate multiple augmented versions
    print("\nGenerating 3 augmented versions at intensity 0.3:")
    
    for i in range(3):
        augmented = imaen_simple(insulin_a, intensity=0.3)
        print(f"\nVersion {i+1}:")
        print(f"  {''.join(augmented)}")
        differences = sum(1 for a, b in zip(insulin_a, augmented) if a != b)
        print(f"  Differences: {differences}/{len(insulin_a)}")


def demo_multiple_augmentations():
    """Show multiple augmentations of the same sequence."""
    print_section_header("Demo 6: Multiple Augmentations (Stochastic Diversity)")
    
    sequence = list("MKFLKFSLLTAVLLSVVFAFSSCGDDPK")
    intensity = 0.3
    
    print(f"\nOriginal: {''.join(sequence)}")
    print(f"\nGenerating 5 random augmentations at intensity {intensity}:")
    
    augmentations = []
    for i in range(5):
        augmented = imaen_simple(sequence, intensity=intensity)
        augmentations.append(''.join(augmented))
        differences = sum(1 for a, b in zip(sequence, augmented) if a != b)
        print(f"\n  Aug {i+1}: {augmentations[-1]}")
        print(f"         Changes: {differences} positions")
    
    # Check diversity
    unique = len(set(augmentations))
    print(f"\nDiversity: {unique} unique sequences out of 5 augmentations")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("  IMAEN: Interpretable Molecular Augmentation for Proteins")
    print("  Property-Aware Amino Acid Substitution Demonstration")
    print("=" * 70)
    
    try:
        demo_basic_augmentation()
        demo_property_biases()
        demo_conservative_vs_diverse()
        demo_specific_properties()
        demo_realistic_protein()
        demo_multiple_augmentations()
        
        print("\n" + "=" * 70)
        print("  Demonstration Complete")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • IMAEN creates sequence diversity while preserving properties")
        print("  • Intensity controls the fraction of positions modified")
        print("  • Property bias allows targeted augmentation")
        print("  • Conservative mode ensures biochemical similarity")
        print("  • Stochastic process generates diverse augmentations")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[X] Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
