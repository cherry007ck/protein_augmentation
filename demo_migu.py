"""
Demonstration of MiGu (Molecular Interactions and Geometric Upgrading) augmentation.

This script shows:
1. Basic usage of MiGu augmentation
2. Context-aware substitution examples
3. Interaction pattern preservation
4. Integration with example.py framework
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentations.migu_augmentation import (
    migu_augment,
    analyze_local_context,
    POSITIVE_CHARGED,
    NEGATIVE_CHARGED,
    HYDROPHOBIC,
    AROMATIC,
)


def demo_basic_usage():
    """Demonstrate basic MiGu augmentation usage."""
    print("="*60)
    print("Demo 1: Basic Usage")
    print("="*60)
    
    # Example protein sequence
    seq = list("MKTAYIAKQRQISFVKSHFSRQ")
    print(f"Original sequence: {''.join(seq)}")
    print(f"Length: {len(seq)} residues\n")
    
    # Apply MiGu augmentation with moderate intensity
    aug_seq = migu_augment(seq, substitution_rate=0.3)
    print(f"Augmented sequence: {''.join(aug_seq)}")
    print(f"Length: {len(aug_seq)} residues")
    
    # Count changes
    changes = sum(1 for i in range(len(seq)) if seq[i] != aug_seq[i])
    print(f"Number of substitutions: {changes} ({changes/len(seq)*100:.1f}%)\n")


def demo_context_awareness():
    """Demonstrate context-aware substitution."""
    print("="*60)
    print("Demo 2: Context-Aware Substitution")
    print("="*60)
    
    # Sequence with distinct regions
    seq = list("SSVVVIIISSKKEEEESS")
    #             ^^^^^^     (hydrophobic region)
    #                   ^^^^    (charged region)
    
    print(f"Original: {''.join(seq)}")
    print("\nRegions:")
    print("  Positions 2-7: VVVIII (hydrophobic)")
    print("  Positions 10-13: KKEE (charged)\n")
    
    # Analyze context at different positions
    print("Context analysis:")
    
    # In hydrophobic region
    ctx_hydro = analyze_local_context(seq, 5, window_size=3)
    print(f"  Position 5 (hydrophobic region): {ctx_hydro}")
    
    # In charged region
    ctx_charged = analyze_local_context(seq, 11, window_size=3)
    print(f"  Position 11 (charged region): {ctx_charged}\n")
    
    # Apply augmentation
    aug_seq = migu_augment(seq, substitution_rate=0.4)
    print(f"Augmented: {''.join(aug_seq)}")
    print("\nNote: Substitutions respect local context!\n")


def demo_disulfide_preservation():
    """Demonstrate disulfide bond preservation."""
    print("="*60)
    print("Demo 3: Disulfide Bond Preservation")
    print("="*60)
    
    # Sequence with potential disulfide bond (two cysteines)
    seq = list("MKTCYIAKQRQCSFVK")
    #             ^       ^  (cysteines at positions 3 and 11)
    
    print(f"Original: {''.join(seq)}")
    print("Cysteines at positions 3 and 11 (potential disulfide bond)\n")
    
    print("Running augmentation 5 times with preservation ON:")
    for i in range(5):
        aug_seq = migu_augment(seq, substitution_rate=0.6, preserve_interactions=True)
        c_positions = [i for i, aa in enumerate(aug_seq) if aa == 'C']
        print(f"  {i+1}. {''.join(aug_seq)} - C at positions: {c_positions}")
    
    print("\nNote: Cysteine pairs are preserved!\n")


def demo_salt_bridge_preservation():
    """Demonstrate salt bridge preservation."""
    print("="*60)
    print("Demo 4: Salt Bridge Preservation")
    print("="*60)
    
    # Sequence with potential salt bridge
    seq = list("MGKDDLTKRAAE")
    #             ^^ (K-D pair at positions 2-3)
    
    print(f"Original: {''.join(seq)}")
    print("K-D pair at positions 2-3 (potential salt bridge)\n")
    
    print("Running augmentation 5 times with preservation ON:")
    for i in range(5):
        aug_seq = migu_augment(seq, substitution_rate=0.5, preserve_interactions=True)
        print(f"  {i+1}. {''.join(aug_seq)}")
        
        # Check if charges are maintained
        if seq[2] in POSITIVE_CHARGED and seq[3] in NEGATIVE_CHARGED:
            if aug_seq[2] in POSITIVE_CHARGED and aug_seq[3] in NEGATIVE_CHARGED:
                print("      -> Salt bridge potential preserved!")
    
    print()


def demo_aromatic_cluster():
    """Demonstrate aromatic cluster preservation."""
    print("="*60)
    print("Demo 5: Aromatic Cluster Preservation")
    print("="*60)
    
    # Sequence with aromatic cluster
    seq = list("MGFYFAKQR")
    #             ^^^ (aromatic cluster F-Y-F at positions 2-4)
    
    print(f"Original: {''.join(seq)}")
    print("Aromatic cluster F-Y-F at positions 2-4\n")
    
    print("Running augmentation 5 times:")
    for i in range(5):
        aug_seq = migu_augment(seq, substitution_rate=0.5, preserve_interactions=True)
        cluster = ''.join(aug_seq[2:5])
        aromatic_count = sum(1 for aa in cluster if aa in AROMATIC)
        print(f"  {i+1}. {''.join(aug_seq)} - Cluster: {cluster} ({aromatic_count}/3 aromatic)")
    
    print("\nNote: Aromatic character tends to be preserved!\n")


def demo_intensity_control():
    """Demonstrate intensity parameter."""
    print("="*60)
    print("Demo 6: Intensity Control")
    print("="*60)
    
    seq = list("ACDEFGHIKLMNPQRSTVWY" * 2)
    print(f"Original: {''.join(seq)}")
    print(f"Length: {len(seq)} residues\n")
    
    intensities = [0.1, 0.3, 0.5, 0.7]
    
    for intensity in intensities:
        aug_seq = migu_augment(seq, substitution_rate=intensity)
        changes = sum(1 for i in range(len(seq)) if seq[i] != aug_seq[i])
        print(f"Intensity {intensity:.1f}: {changes:2d} changes ({changes/len(seq)*100:4.1f}%)")
    
    print()


def demo_preservation_toggle():
    """Demonstrate the preserve_interactions parameter."""
    print("="*60)
    print("Demo 7: Interaction Preservation Toggle")
    print("="*60)
    
    seq = list("MKTCYIAKQRQCSFVK")
    print(f"Original: {''.join(seq)}\n")
    
    print("With preservation ON:")
    aug_on = migu_augment(seq, 0.6, preserve_interactions=True)
    print(f"  {''.join(aug_on)}\n")
    
    print("With preservation OFF:")
    aug_off = migu_augment(seq, 0.6, preserve_interactions=False)
    print(f"  {''.join(aug_off)}\n")
    
    print("Note: Preservation ON maintains critical interactions.")
    print("      Preservation OFF allows more aggressive substitutions.\n")


def demo_context_window():
    """Demonstrate context window parameter."""
    print("="*60)
    print("Demo 8: Context Window Size")
    print("="*60)
    
    seq = list("SSVVVIIISSKKEEE")
    print(f"Original: {''.join(seq)}\n")
    
    print("Small window (context_window=1):")
    aug_small = migu_augment(seq, 0.5, context_window=1)
    print(f"  {''.join(aug_small)}")
    print("  -> Considers only immediate neighbors\n")
    
    print("Large window (context_window=5):")
    aug_large = migu_augment(seq, 0.5, context_window=5)
    print(f"  {''.join(aug_large)}")
    print("  -> Considers broader context\n")


def demo_framework_integration():
    """Demonstrate integration with framework interface."""
    print("="*60)
    print("Demo 9: Framework Integration")
    print("="*60)
    
    print("MiGu follows the standard framework interface:")
    print("  migu_augment(sequence: List[str], intensity: float) -> List[str]\n")
    
    seq = list("MKTCYIAK")
    print(f"Example sequence: {''.join(seq)}\n")
    
    print("Can be used in AUGMENTATION_FUNCTIONS list in example.py:")
    print("  - Automatically selected during random augmentation")
    print("  - Compatible with policy-based augmentation (APA)")
    print("  - Works with dataloader augmentation pipeline\n")
    
    # Simulate framework usage
    print("Simulating framework call:")
    augmentation_intensity = 0.3
    result = migu_augment(seq, augmentation_intensity)
    print(f"  Input: {seq}")
    print(f"  Intensity: {augmentation_intensity}")
    print(f"  Output: {result}")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("MiGu (Molecular Interactions and Geometric Upgrading)")
    print("Context-Aware Augmentation for Protein Sequences")
    print("="*60 + "\n")
    
    demo_basic_usage()
    demo_context_awareness()
    demo_disulfide_preservation()
    demo_salt_bridge_preservation()
    demo_aromatic_cluster()
    demo_intensity_control()
    demo_preservation_toggle()
    demo_context_window()
    demo_framework_integration()
    
    print("="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
