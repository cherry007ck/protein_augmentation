"""
Configuration Manager for Protein Augmentation Training
Loads and validates YAML configuration files
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import torch


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    task_type: str
    train_path: str
    valid_path: str
    test_path: str
    additional_test_paths: List[str] = field(default_factory=list)
    max_sequence_length: int = 512


@dataclass
class TaskConfig:
    """Task configuration"""
    type: str
    num_classes: int
    class_names: Optional[List[str]] = None
    label_type: Optional[str] = None  # For homology tasks


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_class: str
    embed_dim: int
    hidden_dim: int
    vocab_size: int = 21
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.3


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    enabled: bool = False
    intensity: float = 0.15
    probability: float = 0.7
    techniques: List[str] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer_class: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    enabled: bool = False
    scheduler_class: str = "ReduceLROnPlateau"
    mode: str = "max"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6
    T_max: Optional[int] = None  # For CosineAnnealingLR
    eta_min: Optional[float] = None


@dataclass
class TrainConfig:
    """Training configuration"""
    num_epochs: int = 15
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    gradient_clipping_enabled: bool = False
    gradient_clipping_max_norm: float = 1.0


@dataclass
class DeviceConfig:
    """Device configuration"""
    gpus: List[int] = field(default_factory=list)
    mixed_precision: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_interval: int = 10
    save_best_model: bool = True
    save_last_model: bool = True
    tensorboard: bool = False
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


class ConfigManager:
    """Manages loading and parsing of YAML configuration files"""
    
    def __init__(self, config_path: str):
        """
        Initialize config manager
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.raw_config = self._load_yaml()
        self._validate_config()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate required fields in config"""
        required_sections = ['dataset', 'task', 'model', 'train', 'optimizer']
        for section in required_sections:
            if section not in self.raw_config:
                raise ValueError(f"Missing required section: {section}")
    
    def get_output_dir(self) -> str:
        """Get output directory"""
        output_dir = self.raw_config.get('output_dir', './training_results/')
        # Expand ~ to home directory
        output_dir = os.path.expanduser(output_dir)
        return output_dir
    
    def get_dataset_config(self) -> DatasetConfig:
        """Parse dataset configuration"""
        ds = self.raw_config['dataset']
        return DatasetConfig(
            name=ds['name'],
            task_type=ds['task_type'],
            train_path=ds['train_path'],
            valid_path=ds['valid_path'],
            test_path=ds['test_path'],
            additional_test_paths=ds.get('additional_test_paths', []),
            max_sequence_length=ds.get('max_sequence_length', 512)
        )
    
    def get_task_config(self) -> TaskConfig:
        """Parse task configuration"""
        task = self.raw_config['task']
        return TaskConfig(
            type=task['type'],
            num_classes=task['num_classes'],
            class_names=task.get('class_names'),
            label_type=task.get('label_type')
        )
    
    def get_model_config(self) -> ModelConfig:
        """Parse model configuration"""
        model = self.raw_config['model']
        return ModelConfig(
            model_class=model['class'],
            embed_dim=model['embed_dim'],
            hidden_dim=model['hidden_dim'],
            vocab_size=model.get('vocab_size', 21),
            num_layers=model.get('num_layers', 1),
            bidirectional=model.get('bidirectional', True),
            dropout=model.get('dropout', 0.3)
        )
    
    def get_augmentation_config(self) -> AugmentationConfig:
        """Parse augmentation configuration"""
        aug = self.raw_config.get('augmentation', {})
        return AugmentationConfig(
            enabled=aug.get('enabled', False),
            intensity=aug.get('intensity', 0.15),
            probability=aug.get('probability', 0.7),
            techniques=aug.get('techniques', [])
        )
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """Parse optimizer configuration"""
        opt = self.raw_config['optimizer']
        return OptimizerConfig(
            optimizer_class=opt.get('class', 'Adam'),
            lr=opt.get('lr', 1e-3),
            weight_decay=opt.get('weight_decay', 0.0),
            betas=opt.get('betas', [0.9, 0.999])
        )
    
    def get_scheduler_config(self) -> Optional[SchedulerConfig]:
        """Parse scheduler configuration"""
        sched = self.raw_config.get('scheduler', {})
        if not sched.get('enabled', False):
            return None
        
        return SchedulerConfig(
            enabled=True,
            scheduler_class=sched.get('class', 'ReduceLROnPlateau'),
            mode=sched.get('mode', 'max'),
            factor=sched.get('factor', 0.5),
            patience=sched.get('patience', 5),
            min_lr=sched.get('min_lr', 1e-6),
            T_max=sched.get('T_max'),
            eta_min=sched.get('eta_min')
        )
    
    def get_train_config(self) -> TrainConfig:
        """Parse training configuration"""
        train = self.raw_config['train']
        early_stop = train.get('early_stopping', {})
        grad_clip = train.get('gradient_clipping', {})
        
        return TrainConfig(
            num_epochs=train.get('num_epochs', 15),
            batch_size=train.get('batch_size', 32),
            num_workers=train.get('num_workers', 0),
            shuffle=train.get('shuffle', True),
            early_stopping_enabled=early_stop.get('enabled', True),
            early_stopping_patience=early_stop.get('patience', 10),
            early_stopping_min_delta=early_stop.get('min_delta', 0.001),
            gradient_clipping_enabled=grad_clip.get('enabled', False),
            gradient_clipping_max_norm=grad_clip.get('max_norm', 1.0)
        )
    
    def get_device_config(self) -> DeviceConfig:
        """Parse device configuration"""
        device = self.raw_config.get('device', {})
        return DeviceConfig(
            gpus=device.get('gpus', []),
            mixed_precision=device.get('mixed_precision', False)
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Parse logging configuration"""
        log = self.raw_config.get('logging', {})
        wandb = log.get('wandb', {})
        
        return LoggingConfig(
            log_interval=log.get('log_interval', 10),
            save_best_model=log.get('save_best_model', True),
            save_last_model=log.get('save_last_model', True),
            tensorboard=log.get('tensorboard', False),
            wandb_enabled=wandb.get('enabled', False),
            wandb_project=wandb.get('project'),
            wandb_entity=wandb.get('entity')
        )
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics to compute"""
        return self.raw_config.get('metrics', ['accuracy'])
    
    def get_criterion(self) -> str:
        """Get loss criterion"""
        return self.raw_config.get('criterion', 'ce')
    
    def get_eval_metric(self) -> str:
        """Get primary evaluation metric"""
        return self.raw_config.get('eval_metric', 'accuracy')
    
    def get_seed(self) -> int:
        """Get random seed"""
        return self.raw_config.get('seed', 42)
    
    def get_device(self) -> torch.device:
        """Get torch device based on config"""
        device_config = self.get_device_config()
        
        if device_config.gpus and torch.cuda.is_available():
            # Use first GPU in list
            return torch.device(f'cuda:{device_config.gpus[0]}')
        else:
            return torch.device('cpu')
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        
        print(f"\nDataset: {self.raw_config['dataset']['name']}")
        print(f"Task: {self.raw_config['task']['type']}")
        print(f"Model: {self.raw_config['model']['class']}")
        print(f"Epochs: {self.raw_config['train']['num_epochs']}")
        print(f"Batch Size: {self.raw_config['train']['batch_size']}")
        print(f"Learning Rate: {self.raw_config['optimizer']['lr']}")
        
        aug = self.get_augmentation_config()
        print(f"Augmentation: {'Enabled' if aug.enabled else 'Disabled'}")
        if aug.enabled:
            print(f"  - Intensity: {aug.intensity}")
            print(f"  - Probability: {aug.probability}")
            print(f"  - Techniques: {len(aug.techniques)}")
        
        device = self.get_device()
        print(f"Device: {device}")
        
        print(f"Metrics: {', '.join(self.get_metrics())}")
        print(f"Primary Metric: {self.get_eval_metric()}")
        
        print("=" * 80)


def load_config(config_name: str) -> ConfigManager:
    """
    Load a config file by name
    
    Args:
        config_name: Name of config file (without .yaml extension) or full path
    
    Returns:
        ConfigManager instance
    """
    if os.path.exists(config_name):
        # Full path provided
        config_path = config_name
    else:
        # Look in configs directory
        config_path = os.path.join('configs', f'{config_name}.yaml')
    
    return ConfigManager(config_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = 'yeast_ppi'
    
    print(f"Loading config: {config_name}")
    config = load_config(config_name)
    config.print_summary()
    
    print("\n\nDetailed Configuration:")
    print(f"Dataset Config: {config.get_dataset_config()}")
    print(f"Task Config: {config.get_task_config()}")
    print(f"Model Config: {config.get_model_config()}")
    print(f"Augmentation Config: {config.get_augmentation_config()}")
    print(f"Training Config: {config.get_train_config()}")
