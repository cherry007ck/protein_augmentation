"""
Config-based Training Script
Train models using YAML configuration files
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import json

from config_manager import load_config
from models import (
    PPIDataset, LocalizationDataset, HomologyDataset,
    ProteinInteractionLSTM, ProteinClassificationLSTM,
    collate_fn_ppi, collate_fn_classification
)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_datasets(dataset_config, task_config, aug_config):
    """Create train/val/test datasets based on config"""
    
    if dataset_config.task_type == 'ppi':
        train_ds = PPIDataset(
            dataset_config.train_path,
            max_length=dataset_config.max_sequence_length,
            augment=aug_config.enabled,
            augmentation_intensity=aug_config.intensity,
            augmentation_prob=aug_config.probability
        )
        val_ds = PPIDataset(dataset_config.valid_path, augment=False)
        test_ds = PPIDataset(dataset_config.test_path, augment=False)
        collate_fn = collate_fn_ppi
        
    elif dataset_config.task_type == 'localization':
        train_ds = LocalizationDataset(
            dataset_config.train_path,
            max_length=dataset_config.max_sequence_length,
            augment=aug_config.enabled,
            augmentation_intensity=aug_config.intensity,
            augmentation_prob=aug_config.probability
        )
        val_ds = LocalizationDataset(dataset_config.valid_path, augment=False)
        test_ds = LocalizationDataset(dataset_config.test_path, augment=False)
        collate_fn = collate_fn_classification
        
    elif dataset_config.task_type == 'homology':
        label_type = task_config.label_type or 'fold'
        train_ds = HomologyDataset(
            dataset_config.train_path,
            max_length=dataset_config.max_sequence_length,
            augment=aug_config.enabled,
            augmentation_intensity=aug_config.intensity,
            augmentation_prob=aug_config.probability,
            label_type=label_type
        )
        val_ds = HomologyDataset(dataset_config.valid_path, augment=False, label_type=label_type)
        test_ds = HomologyDataset(dataset_config.test_path, augment=False, label_type=label_type)
        collate_fn = collate_fn_classification
    else:
        raise ValueError(f"Unknown task type: {dataset_config.task_type}")
    
    return train_ds, val_ds, test_ds, collate_fn


def create_model(model_config, task_config, device):
    """Create model based on config"""
    
    if model_config.model_class == 'ProteinInteractionLSTM':
        model = ProteinInteractionLSTM(
            embed_dim=model_config.embed_dim,
            hidden_dim=model_config.hidden_dim,
            vocab_size=model_config.vocab_size
        )
    elif model_config.model_class == 'ProteinClassificationLSTM':
        model = ProteinClassificationLSTM(
            num_classes=task_config.num_classes,
            embed_dim=model_config.embed_dim,
            hidden_dim=model_config.hidden_dim,
            vocab_size=model_config.vocab_size,
            num_layers=model_config.num_layers
        )
    else:
        raise ValueError(f"Unknown model class: {model_config.model_class}")
    
    return model.to(device)


def create_optimizer(model, optimizer_config):
    """Create optimizer based on config"""
    
    if optimizer_config.optimizer_class == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=tuple(optimizer_config.betas)
        )
    elif optimizer_config.optimizer_class == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            momentum=0.9
        )
    elif optimizer_config.optimizer_class == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config.optimizer_class}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_config, steps_per_epoch=None):
    """Create learning rate scheduler based on config"""
    
    if scheduler_config is None or not scheduler_config.enabled:
        return None
    
    if scheduler_config.scheduler_class == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.mode,
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.min_lr
        )
    elif scheduler_config.scheduler_class == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.T_max,
            eta_min=scheduler_config.eta_min or 0
        )
    elif scheduler_config.scheduler_class == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.patience,
            gamma=scheduler_config.factor
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_config.scheduler_class}")
    
    return scheduler


def create_criterion(criterion_name, task_type):
    """Create loss criterion based on config"""
    
    if criterion_name == 'bce':
        return nn.BCELoss()
    elif criterion_name == 'ce':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def evaluate(model, loader, criterion, device, task_type):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if task_type == 'ppi':
                seq1, seq2, labels = batch  # Fixed: unpack 3 values directly
                seq1, seq2 = seq1.to(device), seq2.to(device)
                labels = labels.to(device)
                outputs = model(seq1, seq2)
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            else:
                seqs, labels = batch
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                predicted = outputs.argmax(dim=1)
            
            total_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def train_epoch(model, loader, optimizer, criterion, device, task_type, 
                train_config, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        if task_type == 'ppi':
            seq1, seq2, labels = batch  # Fixed: unpack 3 values directly
            seq1, seq2 = seq1.to(device), seq2.to(device)
            labels = labels.to(device)
            outputs = model(seq1, seq2)
        else:
            seqs, labels = batch
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping if enabled
        if train_config.gradient_clipping_enabled:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                train_config.gradient_clipping_max_norm
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    return avg_loss


def train_model(config_manager):
    """Main training function"""
    
    # Load all configs
    dataset_config = config_manager.get_dataset_config()
    task_config = config_manager.get_task_config()
    model_config = config_manager.get_model_config()
    aug_config = config_manager.get_augmentation_config()
    opt_config = config_manager.get_optimizer_config()
    sched_config = config_manager.get_scheduler_config()
    train_config = config_manager.get_train_config()
    device = config_manager.get_device()
    
    # Set seed
    set_seed(config_manager.get_seed())
    
    # Print config summary
    config_manager.print_summary()
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds, collate_fn = create_datasets(
        dataset_config, task_config, aug_config
    )
    print(f"Train: {len(train_ds)}, Valid: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_config.batch_size,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers
    )
    
    # Create model
    print(f"\nCreating model: {model_config.model_class}...")
    model = create_model(model_config, task_config, device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, opt_config)
    scheduler = create_scheduler(optimizer, sched_config, len(train_loader))
    
    # Create criterion
    criterion = create_criterion(
        config_manager.get_criterion(),
        task_config.type
    )
    
    # Training loop
    print(f"\nStarting training for {train_config.num_epochs} epochs...")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(train_config.num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            dataset_config.task_type, train_config, epoch, train_config.num_epochs
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, dataset_config.task_type
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{train_config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Check for improvement
        if val_acc > best_val_acc + train_config.early_stopping_min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            if config_manager.get_logging_config().save_best_model:
                output_dir = config_manager.get_output_dir()
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), 
                          os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        
        # Early stopping
        if (train_config.early_stopping_enabled and 
            patience_counter >= train_config.early_stopping_patience):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # Test evaluation
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, dataset_config.task_type
    )
    print(f"Test Acc: {test_acc:.4f}")
    
    # Save results
    output_dir = config_manager.get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'dataset': dataset_config.name,
        'task_type': dataset_config.task_type,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'history': history,
        'config_file': config_manager.config_path
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train models using YAML configuration files"
    )
    parser.add_argument(
        'config',
        nargs='?',  # Make optional
        help='Config file name (without .yaml) or path to config file'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available config files'
    )
    
    args = parser.parse_args()
    
    if args.list_configs:
        configs_dir = 'configs'
        if os.path.exists(configs_dir):
            configs = [f.replace('.yaml', '') for f in os.listdir(configs_dir) 
                      if f.endswith('.yaml')]
            print("Available configurations:")
            for config in sorted(configs):
                print(f"  - {config}")
        else:
            print("No configs directory found")
        return
    
    if not args.config:
        parser.error("config argument is required when not using --list-configs")
    
    # Load config and train
    print(f"Loading configuration: {args.config}")
    config_manager = load_config(args.config)
    
    # Train model
    results = train_model(config_manager)


if __name__ == "__main__":
    main()
