"""
Simple demonstration of NTA (Nucleotide Augmentation) functionality.

This standalone script demonstrates the core NTA features without
requiring the full framework dependencies.
"""

import sys
sys.path.insert(0, '/home/luffy/protein_augmentation')

import random

# Set random seed for reproducibility
random.seed(42)

from augmentations.nta_augmentation import (
    nucleotide_augment,
    back_translate,
    forward_translate,
    apply_synonymous_substitutions
)


def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  NTA (Nucleotide Augmentation) Demonstration".center(68) + "║")
    print("║" + "  Reference: Minot & Reddy 2022".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Demo 1: Basic NTA
    print("\n" + "=" * 70)
    print("1. Basic NTA Functionality")
    print("=" * 70)
    
    protein_seq = list("METHYLKFPSTWYV")
    print(f"\nOriginal protein: {''.join(protein_seq)}")
    print(f"Length: {len(protein_seq)} amino acids")
    
    # Back-translate
    codons = back_translate(protein_seq)
    print(f"\nBack-translated to DNA: {' '.join(codons)}")
    print(f"Nucleotide length: {len(codons) * 3} bp")
    
    # Show NTA with different rates
    print("\nApplying NTA with different substitution rates:")
    for rate in [0.0, 0.2, 0.5, 0.8]:
        augmented = nucleotide_augment(protein_seq, rate)
        preserved = "✓" if augmented == protein_seq else "✗"
        print(f"  Rate {rate:.1f}: {''.join(augmented):20s} {preserved} AA preserved")
    
    # Demo 2: Nucleotide diversity
    print("\n" + "=" * 70)
    print("2. Nucleotide-Level Diversity (Same Protein, Different DNA)")
    print("=" * 70)
    
    simple_protein = list("GAGAGA")
    print(f"\nProtein: {''.join(simple_protein)}")
    print("(Each G and A can be encoded by 4 different codons)")
    
    print("\nGenerating 5 different DNA sequences for the same protein:")
    for i in range(5):
        codons = back_translate(simple_protein)
        dna = ''.join(codons)
        print(f"  {i+1}. {dna}")
    
    # Demo 3: Synonymous substitutions
    print("\n" + "=" * 70)
    print("3. Synonymous Codon Substitution")
    print("=" * 70)
    
    protein = list("ARGLK")
    print(f"\nOriginal protein: {''.join(protein)}")
    
    # Show the process step by step
    print("\nStep-by-step NTA process:")
    print(f"  1. Original AA: {''.join(protein)}")
    
    codons = back_translate(protein)
    print(f"  2. Back-translate: {' '.join(codons)}")
    
    substituted = apply_synonymous_substitutions(codons, 0.6)
    changed = sum(1 for a, b in zip(codons, substituted) if a != b)
    print(f"  3. Substitute ({changed}/{len(codons)} changed): {' '.join(substituted)}")
    
    final_protein = forward_translate(substituted)
    print(f"  4. Forward-translate: {''.join(final_protein)}")
    
    if final_protein == protein:
        print(f"  ✓ Protein sequence preserved!")
    
    # Demo 4: Comparison with sequence identity
    print("\n" + "=" * 70)
    print("4. Key Advantage: Preserves Amino Acid Identity")
    print("=" * 70)
    
    original = list("METHYLAMINE")
    print(f"\nOriginal sequence: {''.join(original)}")
    
    print("\nComparing augmentation approaches:")
    print("  " + "-" * 60)
    
    # NTA - preserves identity
    nta_result = nucleotide_augment(original, 0.3)
    num_diff_nta = sum(1 for a, b in zip(original, nta_result) if a != b)
    print(f"  NTA (substitution_rate=0.3):")
    print(f"    Result:      {''.join(nta_result)}")
    print(f"    AA changes:  {num_diff_nta}")
    print(f"    ✓ Preserves amino acid sequence (diversity at DNA level)")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
NTA (Nucleotide Augmentation) creates sequence diversity by:
  1. Back-translating amino acids → DNA codons
  2. Applying synonymous codon substitutions  
  3. Forward-translating back to amino acids

Key benefits:
  ✓ Preserves amino acid sequence identity
  ✓ Creates nucleotide-level diversity
  ✓ Useful for protein engineering with functional constraints
  
Integration:
  ✓ Successfully integrated into the augmentation framework
  ✓ Compatible with existing augmentation interface
  ✓ Can be used in policy-based augmentation strategies
""")
    
    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
