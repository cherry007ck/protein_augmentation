# Configuration-Based Training System

## Overview

This system allows you to train models using YAML configuration files similar to PEER Benchmark and other modern ML frameworks. All training parameters, model architecture, augmentation settings, and metrics are defined in a single config file.

## Quick Start

### 1. List Available Configurations
```bash
python train_with_config.py --list-configs
```

Output:
```
Available configurations:
  - benchmark
  - remote_homology
  - subcellular_localization
  - yeast_ppi
```

### 2. Train with a Config File
```bash
python train_with_config.py yeast_ppi
```

### 3. Inspect a Config
```bash
python config_manager.py yeast_ppi
```

## Configuration Files

All config files are located in the `configs/` directory:

### Available Configs

1. **yeast_ppi.yaml** - Protein-Protein Interaction task
   - Binary classification
   - 15 epochs, batch size 32
   - LSTM model with 64 embed dim, 128 hidden dim

2. **subcellular_localization.yaml** - Subcellular localization task
   - 10-class classification
   - 15 epochs, batch size 32
   - Larger LSTM: 128 embed dim, 256 hidden dim, 2 layers

3. **remote_homology.yaml** - Remote homology detection
   - 1195-class classification (fold prediction)
   - 15 epochs, batch size 32
   - Larger LSTM: 128 embed dim, 256 hidden dim, 2 layers

4. **benchmark.yaml** - For benchmarking augmentations
   - Configuration for testing all augmentations

## Config File Structure

### Complete Example (yeast_ppi.yaml)

```yaml
# Output directory
output_dir: ./training_results/yeast_ppi/

# Dataset configuration
dataset:
  name: yeast_ppi
  task_type: ppi
  train_path: datasets/yeast_ppi/yeast_ppi_train.lmdb
  valid_path: datasets/yeast_ppi/yeast_ppi_valid.lmdb
  test_path: datasets/yeast_ppi/yeast_ppi_test.lmdb
  additional_test_paths:
    - datasets/yeast_ppi/yeast_ppi_cross_species_test.lmdb
  max_sequence_length: 512

# Task configuration
task:
  type: binary_classification
  num_classes: 2
  class_names: ["No Interaction", "Interaction"]

# Model architecture
model:
  class: ProteinInteractionLSTM
  embed_dim: 64
  hidden_dim: 128
  vocab_size: 21
  bidirectional: true
  num_layers: 1
  dropout: 0.3

# Loss and metrics
criterion: bce  # Binary Cross Entropy
metrics:
  - accuracy
  - precision
  - recall
  - f1_score
  - mcc
  - auroc
eval_metric: accuracy  # Primary metric

# Data augmentation
augmentation:
  enabled: false
  intensity: 0.15
  probability: 0.7
  techniques:
    - crop_random_segment
    - delete_random_residues
    - substitute_random_residues

# Optimizer
optimizer:
  class: Adam
  lr: 1.0e-3
  weight_decay: 0.0
  betas: [0.9, 0.999]

# Learning rate scheduler (optional)
scheduler:
  enabled: false
  class: ReduceLROnPlateau
  mode: max
  factor: 0.5
  patience: 5
  min_lr: 1.0e-6

# Training configuration
train:
  num_epochs: 15
  batch_size: 32
  num_workers: 0
  shuffle: true
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
  gradient_clipping:
    enabled: false
    max_norm: 1.0

# Device configuration
device:
  gpus: [0]  # GPU indices (empty for CPU)
  mixed_precision: false

# Logging
logging:
  log_interval: 10
  save_best_model: true
  save_last_model: true
  tensorboard: false
  wandb:
    enabled: false
    project: protein_augmentation

# Random seed
seed: 42
```

## Key Configuration Sections

### 1. Dataset
```yaml
dataset:
  name: yeast_ppi
  task_type: ppi  # ppi, localization, or homology
  train_path: datasets/yeast_ppi/yeast_ppi_train.lmdb
  valid_path: datasets/yeast_ppi/yeast_ppi_valid.lmdb
  test_path: datasets/yeast_ppi/yeast_ppi_test.lmdb
  max_sequence_length: 512
```

### 2. Model Architecture
```yaml
model:
  class: ProteinInteractionLSTM  # or ProteinClassificationLSTM
  embed_dim: 64
  hidden_dim: 128
  vocab_size: 21
  num_layers: 1
  bidirectional: true
  dropout: 0.3
```

### 3. Augmentation
```yaml
augmentation:
  enabled: true  # Enable/disable augmentation
  intensity: 0.15  # 0.0-1.0
  probability: 0.7  # 0.0-1.0
  techniques:
    - crop_random_segment
    - nucleotide_augment
    - mask_residues
```

### 4. Training Parameters
```yaml
train:
  num_epochs: 15
  batch_size: 32
  early_stopping:
    enabled: true
    patience: 10
  gradient_clipping:
    enabled: true
    max_norm: 1.0
```

### 5. Optimizer & Scheduler
```yaml
optimizer:
  class: Adam
  lr: 1.0e-3
  weight_decay: 0.0

scheduler:
  enabled: true
  class: ReduceLROnPlateau
  patience: 5
```

## Creating Custom Configs

### Example: Create a new config for high-capacity model

```yaml
# configs/yeast_ppi_large.yaml
output_dir: ./training_results/yeast_ppi_large/

dataset:
  name: yeast_ppi
  task_type: ppi
  train_path: datasets/yeast_ppi/yeast_ppi_train.lmdb
  valid_path: datasets/yeast_ppi/yeast_ppi_valid.lmdb
  test_path: datasets/yeast_ppi/yeast_ppi_test.lmdb
  max_sequence_length: 512

task:
  type: binary_classification
  num_classes: 2

model:
  class: ProteinInteractionLSTM
  embed_dim: 128  # Larger
  hidden_dim: 256  # Larger
  vocab_size: 21
  bidirectional: true
  dropout: 0.5  # More dropout

optimizer:
  class: Adam
  lr: 5.0e-4  # Lower learning rate
  weight_decay: 1.0e-4  # Add weight decay

train:
  num_epochs: 30  # More epochs
  batch_size: 16  # Smaller batch
  gradient_clipping:
    enabled: true
    max_norm: 1.0

scheduler:
  enabled: true
  class: ReduceLROnPlateau
  patience: 7

criterion: bce
metrics: [accuracy, precision, recall, f1_score]
eval_metric: f1_score  # Use F1 instead of accuracy

seed: 42
```

Then train:
```bash
python train_with_config.py yeast_ppi_large
```

## Supported Options

### Model Classes
- `ProteinInteractionLSTM` - For PPI tasks
- `ProteinClassificationLSTM` - For classification tasks

### Optimizers
- `Adam`
- `AdamW`
- `SGD`

### Schedulers
- `ReduceLROnPlateau`
- `CosineAnnealingLR`
- `StepLR`

### Criteria (Loss Functions)
- `bce` - Binary Cross Entropy
- `ce` - Cross Entropy
- `mse` - Mean Squared Error

### Metrics
- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `mcc` - Matthews Correlation Coefficient
- `auroc` - Area Under ROC Curve
- `top3_accuracy`
- `top5_accuracy`
- `confusion_matrix`

### Augmentation Techniques
- `crop_random_segment`
- `delete_random_residues`
- `reverse_sequence`
- `shuffle_random_segment`
- `cut_and_shuffle`
- `subsequence_shuffle`
- `insert_random_residues`
- `substitute_random_residues`
- `swap_random_residues`
- `back_translation_substitute`
- `nucleotide_augment`
- `mask_residues`
- `conservative_mask_residues`

## Usage Examples

### Train with Default Config
```bash
python train_with_config.py yeast_ppi
```

### Train with Augmentation
Edit config file to enable:
```yaml
augmentation:
  enabled: true
  intensity: 0.15
  probability: 0.7
```

Or create a new config file:
```bash
cp configs/yeast_ppi.yaml configs/yeast_ppi_aug.yaml
# Edit the file to enable augmentation
python train_with_config.py yeast_ppi_aug
```

### Quick Experiment (3 epochs)
```yaml
# configs/yeast_ppi_quick.yaml
# ... copy from yeast_ppi.yaml and change:
train:
  num_epochs: 3
  batch_size: 32
```

```bash
python train_with_config.py yeast_ppi_quick
```

### Multi-GPU Training
```yaml
device:
  gpus: [0, 1, 2, 3]  # Use 4 GPUs
```

Note: Multi-GPU support is configured but DataParallel/DDP implementation needs to be added to train_with_config.py

## Output Structure

Training results are saved to the configured output directory:

```
training_results/yeast_ppi/
├── best_model.pt              # Best model checkpoint
├── results_20260102_123456.json  # Training results and metrics
└── ...
```

Results JSON format:
```json
{
  "dataset": "yeast_ppi",
  "task_type": "ppi",
  "best_val_acc": 0.5053,
  "best_epoch": 2,
  "test_acc": 0.4975,
  "test_loss": 0.6789,
  "history": {
    "train_loss": [0.6906, 0.6730, ...],
    "val_loss": [0.6800, 0.6650, ...],
    "val_acc": [0.5053, 0.5000, ...]
  },
  "config_file": "configs/yeast_ppi.yaml"
}
```

## Advantages of Config-Based Training

1. **Reproducibility**: All settings in one file
2. **Version Control**: Track experiments via config files
3. **Easy Comparison**: Compare configs side-by-side
4. **No Code Changes**: Modify training without editing code
5. **Documentation**: Config serves as experiment documentation
6. **Sharing**: Share configs instead of command-line arguments

## Tips

1. **Start with Default Configs**: Use provided configs as templates
2. **Name Configs Descriptively**: `yeast_ppi_aug_heavy.yaml` instead of `config1.yaml`
3. **Version Your Configs**: Keep configs in git with meaningful commit messages
4. **Document Changes**: Add comments in YAML files
5. **Test Quickly**: Create `_quick.yaml` versions with 2-3 epochs for testing

## Comparison with Command-Line

### Old Way (Command-Line)
```bash
python train_all_datasets.py --datasets yeast_ppi --epochs 15 --batch-size 32 \
    --augment --augmentation-intensity 0.15 --augmentation-prob 0.7 \
    --output-dir results/exp1
```

### New Way (Config File)
```bash
python train_with_config.py yeast_ppi
```

All parameters in `configs/yeast_ppi.yaml` - cleaner, trackable, reproducible!

## Integration with Existing Scripts

You can still use the old scripts:
- `train_all_datasets.py` - Command-line interface
- `benchmark_augmentations.py` - Augmentation benchmarking
- `train_with_config.py` - **New!** Config-based training

All scripts are compatible and share the same models/datasets.
