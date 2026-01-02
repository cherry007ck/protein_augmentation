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

### 3. BootGen (Bootstrapped Generation)

**Reference**: Anand et al., 2023 - "Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences" (NeurIPS 2023)
([arXiv: 2306.00111](https://arxiv.org/abs/2306.00111))

**Description**: A simplified implementation of BootGen that uses bootstrapped sampling with rank-based weighting to generate high-quality augmented sequences.

**Algorithm**:
1. Generate multiple candidate sequences (bootstrapped sampling)
2. Score each candidate using proxy quality function
3. Select best candidate using rank-based probabilistic selection

**Key Features**:
- ✓ Quality-aware augmentation (proxy scoring)
- ✓ Bootstrapped candidate generation
- ✓ Rank-based selection (favors high-quality sequences)
- ✓ Biochemical property preservation
- ✓ Conservative substitutions within same property groups

**Scoring Components**:
- Amino acid composition similarity (40%)
- Biochemical property distribution similarity (40%)
- Sequence length preservation (20%)

**Usage**:
```python
from augmentations.bootgen import bootgen_augment

sequence = ['M', 'E', 'T', 'H', 'Y', 'L', 'A', 'M', 'I', 'N', 'E']

# Low intensity: minimal changes, high quality
aug_conservative = bootgen_augment(sequence, intensity=0.2)

# Moderate intensity: balanced diversity and quality
aug_moderate = bootgen_augment(sequence, intensity=0.5)

# High intensity: more aggressive changes
aug_aggressive = bootgen_augment(sequence, intensity=0.8)

## Integration with Framework

All augmentations are compatible with the framework interface in `example.py`:

```python
# Import all augmentation functions
from example import AUGMENTATION_FUNCTIONS

# Framework now has 15 augmentation techniques
print(len(AUGMENTATION_FUNCTIONS))  # 15

# Random selection during training
import random
aug_func = random.choice(AUGMENTATION_FUNCTIONS)
augmented = aug_func(sequence, intensity=0.3)
```

## Augmentation Summary

| # | Technique | Type | Preserves Identity |
|---|-----------|------|--------------------|
| 1-10 | Original APA techniques | Sequence-level | No |
| 11 | NTA | Nucleotide-level | Yes (AA level) |
| 12 | Residue Masking (MLM) | Masking | No |
| 13 | Conservative Masking | Masking | No |
| 14 | BootGen | Quality-aware generation | Partial (high similarity) |

## Testing

Run the test suites:
```bash
python3 test_nta.py
python3 test_residue_masking.py
python3 test_bootgen.py
```

Run the demonstrations:
```bash
python3 demo_nta_simple.py
python3 demo_bootgen.py
```

## Future Augmentation Techniques

Additional techniques from `research_papers/` that could be integrated:
- IMAEN: Interpretable Molecular Augmentation Encoding Networks
- Spider: Structure-based augmentation
- RSA: Retrieved Sequence Augmentation
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
