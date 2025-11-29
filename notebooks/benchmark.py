"""
Modular Benchmarking Script for Transfer Learning on Plant Village Dataset
============================================================================
Compares pretrained models (AlexNet, ResNet50, VGG16) on the augmented
Plant Village binary classification dataset using PyTorch.

The script uses a modular architecture to easily add new models and
ensures fair comparison through consistent transfer learning setup.

MEMORY OPTIMIZATION (NO DISK CACHING):
---------------------------------------
This version uses TensorFlow's native data pipeline for memory efficiency:
- Leverages TensorFlow's optimized data loading and augmentation
- Converts TF tensors to PyTorch on-the-fly during training
- No intermediate disk or full RAM storage required
- Uses IterableDataset for streaming from TensorFlow
- TensorFlow GPU memory disabled (PyTorch uses GPU instead)

Memory settings:
- batch_size: Controls memory per batch (default: 32)
- num_workers: Set to 0 for IterableDataset (default: 0)
- prefetch_size: TensorFlow prefetch buffer (default: 2)

To further reduce memory:
1. Decrease batch_size in TrainingConfig (default: 32 → 16)
2. Set num_workers=0 (required for IterableDataset)
3. Decrease prefetch_size (default: 2 → 1)

Requirements:
- No disk space needed
- Minimal RAM usage (only active batches)
"""

import os
import time
import json
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import timm

import tensorflow as tf
import tensorflow_datasets as tfds

# Configure TensorFlow to use minimal memory
tf.config.set_visible_devices([], 'GPU')  # Disable GPU for TensorFlow (PyTorch will use it)
# Or if you need TF GPU, set memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"TensorFlow GPU configuration: {e}")


# ============================================================================
# Configuration and Data Classes
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # Must be 0 for IterableDataset
    pin_memory: bool = True
    early_stopping_patience: int = 3
    seed: int = 42
    # TensorFlow prefetch settings
    tf_prefetch_size: int = 2  # TensorFlow prefetch buffer size
    # Model-specific batch sizes for memory optimization
    model_batch_sizes: Dict[str, int] = None
    # Gradient accumulation for large models
    gradient_accumulation_steps: int = 1
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_name: str
    train_time: float
    inference_time: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float]
    val_accuracies: List[float]
    best_epoch: int
    # ROC curve data
    roc_fpr: List[float]  # False Positive Rate
    roc_tpr: List[float]  # True Positive Rate
    roc_thresholds: List[float]  # Decision thresholds


# ============================================================================
# Dataset and Data Loading
# ============================================================================

class TFDatasetIterableWrapper(IterableDataset):
    """Wraps TensorFlow Dataset as PyTorch IterableDataset (zero-copy streaming)."""
    
    def __init__(self, tf_dataset, transform=None, length=None):
        """
        Args:
            tf_dataset: TensorFlow dataset with (image, label) tuples
            transform: Optional PyTorch transforms
            length: Approximate dataset length (for progress tracking)
        """
        self.tf_dataset = tf_dataset
        self.transform = transform
        self.length = length
    
    def __iter__(self):
        """Iterate through TensorFlow dataset and yield PyTorch tensors."""
        for image_np, label_np in tfds.as_numpy(self.tf_dataset):
            # Convert numpy to PIL Image for transforms
            from PIL import Image
            image = Image.fromarray(image_np)
            
            if self.transform:
                image = self.transform(image)
            
            label = int(label_np)
            yield image, label
    
    def __len__(self):
        """Approximate length (may not be exact for augmented datasets)."""
        return self.length if self.length is not None else 0


def load_plant_village_data(config: TrainingConfig, max_samples=None):
    """
    Load and prepare Plant Village dataset with augmentation (memory-efficient).
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("\n" + "="*70)
    print("Loading Plant Village Dataset (Memory-Efficient Mode)")
    print("="*70)
    
    # Load TensorFlow dataset (only has 'train' split)
    plant_village_data, info = tfds.load(
        "plant_village", with_info=True, as_supervised=True
    )
    
    # Create binary labels (0=healthy, 1=diseased)
    label_names = info.features["label"].names
    binary_lookup = np.array(
        [0 if name.split("___", 1)[-1].lower() == "healthy" else 1
         for name in label_names],
        dtype=np.int32,
    )
    binary_lookup_tf = tf.constant(binary_lookup)
    
    def to_binary_label(image, label):
        label = tf.cast(label, tf.int32)
        binary_label = tf.gather(binary_lookup_tf, label)
        return image, binary_label
    
    # Apply binary labeling to the train split
    full_train_ds = plant_village_data["train"].map(
        to_binary_label, num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Split the dataset: 70% train, 15% validation, 15% test
    total_samples = info.splits["train"].num_examples
    train_size = int(0.70 * total_samples)
    val_size = int(0.15 * total_samples)
    # test_size is the remainder
    
    print(f"Total samples: {total_samples}")
    print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {total_samples - train_size - val_size}")
    
    # Create splits
    train_ds = full_train_ds.take(train_size)
    remaining = full_train_ds.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    
    # Count healthy vs diseased for augmentation (from training split)
    healthy_count = 0
    diseased_count = 0
    
    for _, label in tfds.as_numpy(train_ds):
        if int(label) == 0:
            healthy_count += 1
        else:
            diseased_count += 1
    
    print(f"Original training set: {healthy_count} healthy, {diseased_count} diseased")
    
    # Recreate train_ds after counting (since we consumed it)
    train_ds = full_train_ds.take(train_size)
    
    # Apply augmentation to balance classes
    healthy_ds = train_ds.filter(lambda _, lbl: tf.equal(lbl, 0))
    diseased_ds = train_ds.filter(lambda _, lbl: tf.equal(lbl, 1))
    
    def augment_healthy(image, label):
        image_f = tf.image.convert_image_dtype(image, tf.float32)
        image_f = tf.image.random_flip_left_right(image_f)
        image_f = tf.image.random_flip_up_down(image_f)
        image_f = tf.image.rot90(
            image_f, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        )
        image_f = tf.image.random_saturation(image_f, 0.8, 1.25)
        image_f = tf.image.random_hue(image_f, 0.05)
        image_f = tf.image.random_brightness(image_f, 0.12)
        image_f = tf.image.random_contrast(image_f, 0.8, 1.25)
        image_f = tf.clip_by_value(image_f, 0.0, 1.0)
        image_aug = tf.image.convert_image_dtype(image_f, tf.uint8)
        return image_aug, label
    
    def augment_diseased(image, label):
        image_f = tf.image.convert_image_dtype(image, tf.float32)
        
        def augmented():
            aug = tf.image.random_flip_left_right(image_f)
            aug = tf.image.rot90(
                aug, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            )
            aug = tf.image.random_contrast(aug, 0.9, 1.1)
            aug = tf.image.random_brightness(aug, 0.08)
            aug = tf.image.random_hue(aug, 0.03)
            aug = tf.clip_by_value(aug, 0.0, 1.0)
            return tf.image.convert_image_dtype(aug, tf.uint8)
        
        def original():
            return image
        
        choice = tf.random.uniform([], 0.0, 1.0)
        return tf.cond(choice > 0.5, augmented, original), label
    
    # Augment healthy samples to match diseased count
    import math
    healthy_multiplier = max(1, math.ceil(diseased_count / healthy_count) - 1)
    
    augmented_healthy = [healthy_ds]
    for _ in range(healthy_multiplier):
        augmented_healthy.append(
            healthy_ds.map(augment_healthy, num_parallel_calls=tf.data.AUTOTUNE)
        )
    
    healthy_augmented_ds = augmented_healthy[0]
    for ds in augmented_healthy[1:]:
        healthy_augmented_ds = healthy_augmented_ds.concatenate(ds)
    
    healthy_augmented_ds = healthy_augmented_ds.shuffle(4096)
    diseased_augmented_ds = diseased_ds.map(
        augment_diseased, num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Combine augmented datasets and add TensorFlow prefetching
    augmented_train_ds = healthy_augmented_ds.concatenate(diseased_augmented_ds)
    augmented_train_ds = augmented_train_ds.shuffle(8192).prefetch(config.tf_prefetch_size)
    val_ds = val_ds.prefetch(config.tf_prefetch_size)
    test_ds = test_ds.prefetch(config.tf_prefetch_size)
    
    print(f"Augmented training set: ~{healthy_count * (healthy_multiplier + 1)} healthy, {diseased_count} diseased")
    
    # Calculate approximate dataset sizes
    approx_train_size = healthy_count * (healthy_multiplier + 1) + diseased_count
    approx_val_size = val_size
    approx_test_size = total_samples - train_size - val_size
    
    # Define PyTorch transforms (ImageNet normalization for pretrained models)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create PyTorch IterableDatasets (streaming, no disk/memory caching)
    print("\nCreating streaming datasets (no disk caching)...")
    train_dataset = TFDatasetIterableWrapper(
        augmented_train_ds, transform=train_transform, length=approx_train_size
    )
    val_dataset = TFDatasetIterableWrapper(
        val_ds, transform=val_transform, length=approx_val_size
    )
    test_dataset = TFDatasetIterableWrapper(
        test_ds, transform=val_transform, length=approx_test_size
    )
    
    # Create data loaders (num_workers must be 0 for IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,  # IterableDataset requires num_workers=0
        pin_memory=config.pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=config.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=config.pin_memory,
    )
    
    print("\nDataLoaders created (streaming mode):")
    print(f"  Train: ~{len(train_dataset)} samples, ~{len(train_dataset) // config.batch_size} batches")
    print(f"  Val:   ~{len(val_dataset)} samples, ~{len(val_dataset) // config.batch_size} batches")
    print(f"  Test:  ~{len(test_dataset)} samples, ~{len(test_dataset) // config.batch_size} batches")
    print("\nMemory optimization enabled (zero-copy streaming):")
    print("  - No disk caching (zero disk space usage)")
    print("  - Streaming directly from TensorFlow")
    print("  - Only active batches in RAM")
    print("  - num_workers=0 (required for IterableDataset)")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Model Factory - Modular Model Creation
# ============================================================================

def create_alexnet(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create AlexNet with transfer learning."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze early layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace final classifier layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model


def create_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create ResNet50 with transfer learning."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze early layers (unfreeze last residual block for fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 (last residual block) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_vgg16(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create VGG16 with transfer learning."""
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace final classifier layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model


def create_inception_v4(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create InceptionV4 with transfer learning using timm."""
    model = timm.create_model(
        'inception_v4',
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Freeze early layers (keep only final layers trainable)
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier head
    if hasattr(model, 'classif'):
        for param in model.classif.parameters():
            param.requires_grad = True
    elif hasattr(model, 'last_linear'):
        for param in model.last_linear.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model


def create_xception(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create Xception with transfer learning using timm."""
    model = timm.create_model(
        'xception',
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier head
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model


def create_convnext_base(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create ConvNeXt Base with transfer learning."""
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace final classifier layer
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)
    
    return model


# Model registry for easy addition of new models
MODEL_REGISTRY: Dict[str, Callable[[int, bool], nn.Module]] = {
    "AlexNet": create_alexnet,
    "ResNet50": create_resnet50,
    "VGG16": create_vgg16,
    "InceptionV4": create_inception_v4,
    "Xception": create_xception,
    "ConvNeXt_Base": create_convnext_base,
}


# ============================================================================
# Training and Evaluation
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    use_amp: bool = True,
    accumulation_steps: int = 1,
) -> Tuple[float, float]:
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Mixed precision forward pass
        if use_amp and device == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Normalize loss for accumulation
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * images.size(0) * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Clear cache periodically for large models
        if device == "cuda" and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Handle any remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Validate the model with mixed precision."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Mixed precision forward pass
            if use_amp and device == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Clear cache periodically
            if device == "cuda" and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_name: str,
) -> Tuple[nn.Module, Dict[str, List[float]], int]:
    """
    Train a model with early stopping, gradient accumulation, and mixed precision.
    
    Returns:
        Tuple of (trained_model, history_dict, best_epoch)
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    device = config.device
    model = model.to(device)
    
    # Memory optimization: enable gradient checkpointing for large models (when supported)
    if hasattr(model, 'set_grad_checkpointing'):
        try:
            model.set_grad_checkpointing(enable=True)
        except AssertionError:
            print(f"Warning: Gradient checkpointing not supported for {model_name}. Skipping.")
    
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_model_state = None
    
    print("Memory optimizations enabled:")
    print(f"  - Mixed Precision (AMP): {config.use_amp and device == 'cuda'}")
    print(f"  - Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    if device == "cuda":
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    for epoch in range(config.num_epochs):
        start_time = time.time()
        
        # Clear cache before epoch
        if device == "cuda":
            torch.cuda.empty_cache()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=config.use_amp,
            accumulation_steps=config.gradient_accumulation_steps,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            use_amp=config.use_amp,
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        
        # Memory usage logging
        if device == "cuda":
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Epoch [{epoch+1}/{config.num_epochs}] ({epoch_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"GPU Mem: {mem_allocated:.2f}/{mem_reserved:.2f} GB")
        else:
            print(f"Epoch [{epoch+1}/{config.num_epochs}] ({epoch_time:.2f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if best_model_state is None or val_loss < min(history['val_loss'][:-1] + [float('inf')]):
            best_model_state = model.state_dict().copy()
        
        # Early stopping
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return model, history, early_stopping.best_epoch


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set with mixed precision.
    
    Returns:
        Tuple of (y_true, y_pred, y_probs)
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            
            # Mixed precision inference
            if use_amp and device == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
            else:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # Get probabilities and validate for NaN/inf
            probs_pos = probs[:, 1].cpu().numpy()
            # Check for NaN/inf and replace if needed
            if np.any(np.isnan(probs_pos)) or np.any(np.isinf(probs_pos)):
                print(f"Warning: Invalid probabilities in batch {batch_idx}, cleaning...")
                probs_pos = np.nan_to_num(probs_pos, nan=0.5, posinf=1.0, neginf=0.0)
            y_probs.extend(probs_pos)
            
            # Clear cache periodically
            if device == "cuda" and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics including ROC curve data."""
    from sklearn.metrics import roc_curve
    
    # Validate and clean y_probs (handle NaN/inf values)
    if np.any(np.isnan(y_probs)) or np.any(np.isinf(y_probs)):
        print("Warning: NaN or inf values detected in probabilities. Cleaning...")
        # Replace NaN with 0.5 (uncertain) and clip inf values
        y_probs = np.nan_to_num(y_probs, nan=0.5, posinf=1.0, neginf=0.0)
        y_probs = np.clip(y_probs, 0.0, 1.0)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_probs),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'roc_fpr': fpr.tolist(),
        'roc_tpr': tpr.tolist(),
        'roc_thresholds': thresholds.tolist(),
    }


# ============================================================================
# Benchmarking Pipeline
# ============================================================================

def benchmark_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainingConfig,
    save_model: bool = True,
    output_dir: str = "data/output",
) -> ModelMetrics:
    """
    Benchmark a single model.
    
    Args:
        model_name: Name of the model to benchmark
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Training configuration
        save_model: Whether to save the trained model
        output_dir: Directory to save models
    
    Returns:
        ModelMetrics object with all metrics
    """
    print(f"\n{'#'*70}")
    print(f"# Benchmarking {model_name}")
    print(f"{'#'*70}")
    
    # Clear GPU cache before starting
    if config.device == "cuda":
        torch.cuda.empty_cache()
    
    # Create model
    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(num_classes=2, pretrained=True)
    
    # Train model
    train_start = time.time()
    trained_model, history, best_epoch = train_model(
        model, train_loader, val_loader, config, model_name
    )
    train_time = time.time() - train_start
    
    # Evaluate on test set
    print(f"\nEvaluating {model_name} on test set...")
    inference_start = time.time()
    y_true, y_pred, y_probs = evaluate_model(
        trained_model, test_loader, config.device, use_amp=config.use_amp
    )
    inference_time = time.time() - inference_start
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Training Time:   {train_time:.2f}s")
    print(f"  Inference Time:  {inference_time:.2f}s")
    
    # Save model if requested
    if save_model:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pth")
        
        # Save model state dict and training info
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_name': model_name,
            'config': asdict(config),
            'metrics': metrics,
            'history': history,
            'best_epoch': best_epoch,
            'train_time': train_time,
            'inference_time': inference_time,
        }, model_path)
        print(f"  Model saved to: {model_path}")
    
    # Clear memory after model
    del trained_model
    if config.device == "cuda":
        torch.cuda.empty_cache()
    
    return ModelMetrics(
        model_name=model_name,
        train_time=train_time,
        inference_time=inference_time,
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1'],
        auc_roc=metrics['auc_roc'],
        confusion_matrix=metrics['confusion_matrix'],
        train_losses=history['train_loss'],
        val_losses=history['val_loss'],
        train_accuracies=history['train_acc'],
        val_accuracies=history['val_acc'],
        best_epoch=best_epoch,
        roc_fpr=metrics['roc_fpr'],
        roc_tpr=metrics['roc_tpr'],
        roc_thresholds=metrics['roc_thresholds'],
    )


def run_benchmark(
    model_names: List[str],
    config: TrainingConfig,
    max_samples: int = None,
) -> List[ModelMetrics]:
    """
    Run benchmark on multiple models.
    
    Args:
        model_names: List of model names to benchmark
        config: Training configuration
        max_samples: Optional limit on dataset size (for faster testing)
    
    Returns:
        List of ModelMetrics for each model
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    print(f"\n{'='*70}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Models: {', '.join(model_names)}")
    
    # Load data once
    train_loader, val_loader, test_loader = load_plant_village_data(
        config, max_samples=max_samples
    )
    
    # Benchmark each model
    results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            print(f"Warning: Model '{model_name}' not found in registry. Skipping.")
            continue
        
        metrics = benchmark_model(
            model_name, train_loader, val_loader, test_loader, config
        )
        results.append(metrics)
    
    return results


# ============================================================================
# Results Visualization and Export
# ============================================================================

def plot_training_curves(results: List[ModelMetrics], save_path: str = None):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    for result in results:
        epochs = range(1, len(result.train_losses) + 1)
        axes[0].plot(epochs, result.train_losses, label=f"{result.model_name} (Train)", linestyle='--')
        axes[0].plot(epochs, result.val_losses, label=f"{result.model_name} (Val)")
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    for result in results:
        epochs = range(1, len(result.train_accuracies) + 1)
        axes[1].plot(epochs, result.train_accuracies, label=f"{result.model_name} (Train)", linestyle='--')
        axes[1].plot(epochs, result.val_accuracies, label=f"{result.model_name} (Val)")
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(results: List[ModelMetrics], save_path: str = None):
    """Plot comparison of key metrics across models."""
    model_names = [r.model_name for r in results]
    metrics_to_plot = {
        'Accuracy': [r.accuracy for r in results],
        'Precision': [r.precision for r in results],
        'Recall': [r.recall for r in results],
        'F1-Score': [r.f1_score for r in results],
        'AUC-ROC': [r.auc_roc for r in results],
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.15
    multiplier = 0
    
    for metric_name, values in metrics_to_plot.items():
        offset = width * multiplier
        ax.bar(x + offset, values, width, label=metric_name)
        multiplier += 1
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    
    plt.show()


def plot_confusion_matrices(results: List[ModelMetrics], save_path: str = None):
    """Plot confusion matrices for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        cm = np.array(result.confusion_matrix)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{result.model_name}\nConfusion Matrix')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Healthy', 'Diseased'])
        ax.set_yticklabels(['Healthy', 'Diseased'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()


def plot_roc_curves(results: List[ModelMetrics], save_path: str = None):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in results:
        ax.plot(result.roc_fpr, result.roc_tpr, 
               label=f"{result.model_name} (AUC = {result.auc_roc:.4f})",
               linewidth=2)
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def save_results(results: List[ModelMetrics], output_dir: str = "data/output"):
    """Save benchmark results to JSON and create visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Create visualizations
    plot_training_curves(
        results,
        save_path=os.path.join(output_dir, f"training_curves_{timestamp}.png")
    )
    plot_metrics_comparison(
        results,
        save_path=os.path.join(output_dir, f"metrics_comparison_{timestamp}.png")
    )
    plot_confusion_matrices(
        results,
        save_path=os.path.join(output_dir, f"confusion_matrices_{timestamp}.png")
    )
    plot_roc_curves(
        results,
        save_path=os.path.join(output_dir, f"roc_curves_{timestamp}.png")
    )
    
    # Create summary table
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'Train(s)':<10} {'Infer(s)':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.model_name:<15} {r.accuracy:<8.4f} {r.precision:<8.4f} "
              f"{r.recall:<8.4f} {r.f1_score:<8.4f} {r.auc_roc:<8.4f} "
              f"{r.train_time:<10.2f} {r.inference_time:<10.2f}")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main benchmarking pipeline."""
    # Model-specific batch sizes (optimized for memory)
    model_batch_sizes = {
        "AlexNet": 64,
        "ResNet50": 32,
        "VGG16": 32,
        "InceptionV4": 16,  # Reduced for large model
        "Xception": 16,     # Reduced for large model
        "ConvNeXt_Base": 32,  # Similar to ResNet50
    }
    
    # Configuration
    config = TrainingConfig(
        batch_size=32,  # Default, will be overridden per model
        num_epochs=5,
        learning_rate=0.001,
        weight_decay=1e-4,
        early_stopping_patience=2,
        num_workers=0,  # Must be 0 for IterableDataset
        tf_prefetch_size=2,  # TensorFlow prefetch buffer
        model_batch_sizes=model_batch_sizes,
        gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps for large models
        use_amp=True,  # Enable mixed precision training
    )
    
    # For systems with VERY limited memory (< 8GB GPU), use:
    # config.batch_size = 8
    # config.gradient_accumulation_steps = 4
    # model_batch_sizes = {"AlexNet": 16, "ResNet50": 8, "VGG16": 8, "InceptionV4": 4, "Xception": 4}
    
    # Models to benchmark
    # models_to_test = ["AlexNet", "ResNet50", "VGG16", "InceptionV4", "Xception"]
    # models_to_test = ["InceptionV4", "Xception"]
    models_to_test = ["ConvNeXt_Base"]
    
    # Run benchmark with model-specific batch sizes
    results = []
    for model_name in models_to_test:
        # Use model-specific batch size if available
        if config.model_batch_sizes and model_name in config.model_batch_sizes:
            original_batch_size = config.batch_size
            config.batch_size = config.model_batch_sizes[model_name]
            print(f"\n{'='*70}")
            print(f"Using batch size {config.batch_size} for {model_name}")
            print(f"{'='*70}")
        
        # Load data with current batch size
        train_loader, val_loader, test_loader = load_plant_village_data(config)
        
        # Benchmark this model
        try:
            metrics = benchmark_model(
                model_name, train_loader, val_loader, test_loader, config,
                save_model=True, output_dir="data/output"
            )
            results.append(metrics)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ OOM Error for {model_name}. Try reducing batch_size or enabling gradient accumulation.")
                print(f"   Current batch_size: {config.batch_size}")
                print(f"   Suggestion: batch_size={config.batch_size // 2}, gradient_accumulation_steps={config.gradient_accumulation_steps * 2}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        # Restore original batch size
        if config.model_batch_sizes and model_name in config.model_batch_sizes:
            config.batch_size = original_batch_size
    
    # Save and visualize results
    if results:
        save_results(results)
        print("\n✓ Benchmarking complete!")
    else:
        print("\n❌ No models completed successfully!")


if __name__ == "__main__":
    main()
