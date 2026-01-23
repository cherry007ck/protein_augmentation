# Protein Augmentations Package

This package contains augmentation techniques for protein sequences, designed to enhance protein predictive models through data augmentation.

## Available Augmentations

### 1. NTA (Nucleotide Augmentation)

**Reference**: Minot & Reddy, 2022 - "Nucleotide augmentation for machine learning-guided protein engineering"
([DOI: 10.1093/bioadv/vbac094](https://doi.org/10.1093/bioadv/vbac094))

**Description**: Creates sequence diversity at the nucleotide level while preserving amino acid identity.

**Algorithm**:
1. Back-translate amino acids to DNA codons
2. Apply synonymous codon substitutions
3. Forward-translate back to amino acids

**Key Features**:
- ✓ Preserves amino acid sequence identity
- ✓ Creates nucleotide-level diversity
- ✓ Useful for protein engineering with functional constraints
- ✓ Biologically inspired (leverages genetic code degeneracy)

**Usage**:
```python
from augmentations.nta_augmentation import nucleotide_augment

# Augment a protein sequence
sequence = ['M', 'E', 'T', 'H', 'Y', 'L']
substitution_rate = 0.3  # 30% of codons will be substituted

augmented_seq = nucleotide_augment(sequence, substitution_rate)
# Result: Same amino acids, different nucleotide encoding
```

### 2. Residue Masking

**Reference**: Inspired by BERT-style masking (Devlin et al. 2019), ProtBERT (Elnaggar et al. 2020), and ESM (Rives et al. 2021)

**Description**: Masks random residues using MLM-style strategy, replacing them with mask tokens, random amino acids, or leaving them unchanged.

**Variants**:

#### MLM-Style Masking (`mask_residues`)
Follows the BERT masking strategy:
- 80% of selected positions → mask token ('X')
- 10% of selected positions  → random amino acid
- 10% of selected positions → unchanged

#### Simple Masking (`simple_mask_residues`)
All selected positions → mask token ('X')

#### Conservative Masking (`conservative_mask_residues`)
Selected positions → chemically similar amino acid from the same group:
- Hydrophobic: A, V, I, L, M, F, W, P
- Hydrophilic: S, T, N, Q
- Charged positive: K, R, H
- Charged negative: D, E
- Special: C, G, Y

**Key Features**:
- ✓ Simple and effective
- ✓ Proven in protein language models
- ✓ Creates denoising-style augmentation
- ✓ Encourages robust representations

**Usage**:
```python
from augmentations.residue_masking import mask_residues, conservative_mask_residues

sequence = ['M', 'E', 'T', 'H', 'Y', 'L', 'A', 'M', 'I', 'N', 'E']

# MLM-style masking
masked_mlm = mask_residues(sequence, masking_rate=0.15)
# Result: ~15% masked with 80/10/10 strategy

# Conservative masking
masked_cons = conservative_mask_residues(sequence, masking_rate=0.15)
# Result: ~15% replaced with chemically similar AAs
```

### 3. Spider Augmentation

**Reference**: "A Deep Learning Approach with Data Augmentation to Predict Novel Spider Neurotoxic Peptides"

**Description**: Creates sequence diversity through random amino acid substitution and insertion operations.

**Algorithm**:
1. Randomly substitute amino acids with other amino acids
2. Randomly insert new amino acids at random positions

**Key Features**:
- ✓ Simple and effective sequence-level augmentation
- ✓ Increases sequence diversity significantly
- ✓ Can increase sequence length (due to insertions)
- ✓ Biology-inspired (originally for neurotoxic peptide prediction)

**Usage**:
```python
from augmentations.spider_augmentation import spider_augment

# Augment a protein sequence
sequence = ['M', 'E', 'T', 'H', 'Y', 'L', 'A', 'M', 'I', 'N', 'E']
intensity = 0.3  # Controls both substitution and insertion rates

augmented_seq = spider_augment(sequence, intensity)
# Result: Some AAs substituted, some new AAs inserted
# Length may increase due to insertions
```

**Note**: Original paper includes BLAST homology filtering (E-value threshold 1×10⁻⁵) to select biologically plausible sequences. This implementation provides the core augmentation without filtering for framework compatibility.
### 3. NaNa (Novel Augmentation of New Node Attributes)

**Reference**: "NaNa and MiGu: Semantic Data Augmentation Techniques to Enhance Protein Classification in Graph Neural Networks"

**Description**: Semantic augmentation that substitutes amino acids with biophysically and structurally similar alternatives, preserving key molecular properties.

**Algorithm**:
1. Calculate multi-dimensional similarity (hydrophobicity, charge, size, secondary structure propensities)
2. Select positions for substitution based on intensity
3. Substitute with property-preserving alternatives

**Key Features**:
- ✓ Preserves molecular biophysical properties (hydrophobicity, charge, size)
- ✓ Respects secondary structure propensities (α-helix, β-sheet, turn)
- ✓ Multi-dimensional similarity scoring
- ✓ Semantically meaningful augmentation

**Properties Preserved**:
- Hydrophobicity (Kyte-Doolittle scale)
- Charge at pH 7.4
- Van der Waals volume (size)
- α-helix propensity (Chou-Fasman)
- β-sheet propensity (Chou-Fasman)
- Turn propensity (Chou-Fasman)

**Usage**:
```python
from augmentations.nana_augmentation import nana_augment

sequence = ['M', 'E', 'T', 'H', 'I', 'O', 'N', 'I', 'N', 'E']
substitution_rate = 0.3  # 30% of residues will be substituted

augmented_seq = nana_augment(sequence, substitution_rate)
# Result: E→D (both negative), T→S (both polar), I→L/V (both hydrophobic)

# With custom similarity threshold
augmented_seq = nana_augment(
    sequence, 
    substitution_rate=0.3,
    similarity_threshold=0.7,  # More conservative substitutions
    use_groups=True  # Fast lookup using pre-defined groups
)
```

### 4. MiGu (Molecular Interactions and Geometric Upgrading)

**Reference**: "NaNa and MiGu: Semantic Data Augmentation Techniques to Enhance Protein Classification in Graph Neural Networks"

**Description**: Context-aware augmentation that extends NaNa by considering local sequence context and preserving critical molecular interaction patterns.

**Algorithm**:
1. Analyze local sequence context (k-mer window)
2. Identify interaction patterns (disulfide bonds, salt bridges, aromatic clusters)
3. Make context-aware substitutions that preserve critical interactions

**Key Features**:
- ✓ Context-aware substitution based on neighboring residues
- ✓ Preserves disulfide bonds (C-C pairs)
- ✓ Preserves salt bridge potential (K/R-D/E interactions)
- ✓ Maintains aromatic clusters (F/W/Y/H)
- ✓ Respects local hydrophobic/charged environments

**Interaction Patterns Preserved**:
- Disulfide bonds (Cysteine pairs)
- Salt bridges (K/R ↔ D/E)
- Aromatic clusters (π-stacking)
- Hydrophobic cores
- Charged surface regions

**Usage**:
```python
from augmentations.migu_augmentation import migu_augment

sequence = list("MKTCYIAKQRQCSFVK")  # Contains C-C pair (potential disulfide)
substitution_rate = 0.3

# With interaction preservation (default)
augmented_seq = migu_augment(
    sequence, 
    substitution_rate=0.3,
    context_window=3,  # Consider ±3 residues
    preserve_interactions=True  # Preserve critical patterns
)
# Result: C-C pair preserved, context-aware substitutions

# Without preservation (more aggressive)
augmented_seq = migu_augment(
    sequence,
    substitution_rate=0.5,
    preserve_interactions=False
)
```

## Integration with Framework

All augmentations are compatible with the framework interface in `example.py`:

```python
# Import all augmentation functions
from example import AUGMENTATION_FUNCTIONS

# Framework now has 14 augmentation techniques
print(len(AUGMENTATION_FUNCTIONS))  # 14
# Framework now has 15 augmentation techniques
print(len(AUGMENTATION_FUNCTIONS))  # 15

# Random selection during training
import random
aug_func = random.choice(AUGMENTATION_FUNCTIONS)
augmented = aug_func(sequence, intensity=0.3)
```

## Augmentation Summary

| # | Technique | Type | Preserves Identity |
|---|-----------|------|-------------------|
| 1-10 | Original APA techniques | Sequence-level | No |
| 11 | NTA | Nucleotide-level | Yes (AA level) |
| 12 | Residue Masking (MLM) | Masking | No |
| 13 | Conservative Masking | Masking | No |
| 14 | Spider | Substitution + Insertion | No |
| # | Technique | Type | Preserves | Properties Maintained |
|---|-----------|------|-----------|----------------------|
| 1-10 | Original APA techniques | Sequence-level | Varies | N/A |
| 11 | NTA | Nucleotide-level | AA Identity | 100% AA identity |
| 12 | Residue Masking (MLM) | Masking | No | N/A |
| 13 | Conservative Masking | Masking | Chemical class | Property groups |
| 14 | NaNa | Semantic | Properties | Biophysical + Structure |
| 15 | MiGu | Semantic + Context | Interactions | Properties + Interactions |

## Testing

Run the test suites:
```bash
python test_nta.py
python test_residue_masking.py
python test_spider.py
# NTA tests
python3 test_nta.py

# Residue masking tests
python3 test_residue_masking.py

# NanaMigu tests
python3 test_nana.py
python3 test_migu.py
```

Run the demonstrations:
```bash
python demo_nta_simple.py
python demo_spider_simple.py
# NTA demo
python3 demo_nta_simple.py

# NanaMigu demos
python3 demo_nana.py
python3 demo_migu.py
```

## Augmentation Selection Guide

### When to use NTA:
- Preserving exact amino acid sequence is critical
- Working with protein engineering tasks
- Need nucleotide-level diversity

### When to use Residue Masking:
- Training robust protein language models
- Denoising-style augmentation
- Simple and general-purpose augmentation

### When to use NaNa:
- Need property-preserving substitutions
- Maintaining protein stability
- Biophysical constraints matter

### When to use MiGu:
- Preserving structural interactions is important
- Context matters (e.g., binding sites, active sites)
- Working with structured proteins
- Need both properties and interaction preservation

## Future Augmentation Techniques

Additional techniques from `research_papers/` that could be integrated:
- IMAEN: Interpretable Molecular Augmentation Encoding Networks
- RSA: Retrieved Sequence Augmentation
- BootGen: Bootstrapped generation
- PreIS: Protein sequence augmentation

## Contributing

When adding new augmentation techniques:

1. **Follow the interface**: `(sequence: List[str], intensity: float) -> List[str]`
2. **Add type hints**: Use Python typing module
3. **Write docstrings**: Reference the original paper
4. **Create tests**: Verify correctness and integration
5. **Update documentation**: Add to this README

## License

See main repository LICENSE file.
