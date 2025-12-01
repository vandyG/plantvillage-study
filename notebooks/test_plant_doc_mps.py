"""
Test Trained Models on PlantDoc Dataset - MPS Optimized
========================================================
Tests previously trained binary classification models on the PlantDoc dataset.
Optimized for Apple Silicon (MPS), CUDA, and CPU devices.

The PlantDoc dataset contains multi-class labels (different diseases), but our models
perform binary classification (healthy vs diseased). This script:
1. Loads the PlantDoc dataset and converts to binary labels
2. Tests all trained models from the notebooks/models/ directory
3. Provides comprehensive comparison metrics
4. Generates detailed visualizations

Binary Label Mapping:
- "healthy" keywords in folder name → 0 (healthy)
- Any disease name in folder name → 1 (diseased)

Usage:
    python test_plant_doc_mps.py
"""

import os
import time
import json
import glob
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)


# ============================================================================
# Configuration and Data Classes
# ============================================================================

@dataclass
class TestConfig:
    """Configuration for model testing."""
    batch_size: int = 32
    # Support MPS (Apple Silicon), CUDA, and CPU
    device: str = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = False  # Mixed precision inference (CUDA only)


@dataclass
class TestMetrics:
    """Metrics for a single model on test set."""
    model_name: str
    model_path: str
    inference_time: float
    samples_per_second: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    classification_report: str
    # ROC curve data
    roc_fpr: List[float]
    roc_tpr: List[float]
    roc_thresholds: List[float]
    # Per-class metrics
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


# ============================================================================
# Dataset - PlantDoc with Binary Labels
# ============================================================================

class PlantDocBinaryDataset(Dataset):
    """PlantDoc dataset with binary classification (healthy vs diseased)."""
    
    def __init__(self, root_dir: str, split: str = "test", transform=None):
        """
        Args:
            root_dir: Root directory of PlantDoc dataset
            split: 'train' or 'test'
            transform: Optional transform to apply to images
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # Load all images and create binary labels
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            self.class_names.append(class_name)
            
            # Determine binary label: 0 for healthy, 1 for diseased
            # Folders with "leaf" without disease indicators are healthy
            # All others are diseased
            is_healthy = self._is_healthy_class(class_name)
            binary_label = 0 if is_healthy else 1
            
            # Load all images in this class
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), binary_label, class_name))
        
        print(f"\nPlantDoc {split} dataset loaded:")
        print(f"  Total samples: {len(self.samples)}")
        
        # Count samples per binary class
        healthy_count = sum(1 for _, label, _ in self.samples if label == 0)
        diseased_count = sum(1 for _, label, _ in self.samples if label == 1)
        print(f"  Healthy samples: {healthy_count}")
        print(f"  Diseased samples: {diseased_count}")
        print(f"  Unique original classes: {len(self.class_names)}")
    
    def _is_healthy_class(self, class_name: str) -> bool:
        """Determine if a class represents healthy plants."""
        class_lower = class_name.lower()
        
        # Keywords that indicate disease
        disease_keywords = [
            'rust', 'scab', 'spot', 'blight', 'rot', 'mildew', 
            'bacterial', 'virus', 'mosaic', 'mold', 'septoria',
            'spider', 'mites', 'early', 'late', 'gray', 'yellow'
        ]
        
        # If any disease keyword is present, it's diseased
        if any(keyword in class_lower for keyword in disease_keywords):
            return False
        
        # Classes with just plant name and "leaf" are healthy
        # e.g., "Apple leaf", "Tomato leaf", "Cherry leaf"
        if 'leaf' in class_lower:
            # Check if it's a simple "X leaf" pattern
            parts = class_lower.split()
            if len(parts) == 2 and parts[1] == 'leaf':
                return True
        
        # Default to diseased if unsure
        return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# Model Loading
# ============================================================================

def load_trained_model(model_path: str, device: str) -> Tuple[nn.Module, str]:
    """
    Load a trained model from checkpoint.
    
    Returns:
        Tuple of (model, model_name)
    """
    # Extract model name from filename (e.g., "alexnet_best.pth" -> "alexnet")
    filename = os.path.basename(model_path)
    model_name = filename.replace('_best.pth', '').replace('.pth', '')
    
    print(f"\nLoading model: {model_name} from {filename}")
    
    # Create model architecture based on name (num_classes=1 for binary classification with BCEWithLogitsLoss)
    from torchvision import models
    
    lname = model_name.lower()
    
    if lname == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif lname == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, 1)
    elif lname == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, 1)
    elif lname == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif lname == "googlenet":
        model = models.googlenet(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.aux_logits = False
    elif lname == "shufflenet_v2":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif lname == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif lname == "squeezenet1_0":
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Conv2d(512, 1, kernel_size=1)
        model.num_classes = 1
    elif lname == "mnasnet1_0":
        model = models.mnasnet1_0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif "efficientnet" in lname or "inception" in lname or "xception" in lname:
        # Use timm for other models
        model = timm.create_model(lname, pretrained=False, num_classes=1)
    else:
        # Try timm as fallback
        try:
            model = timm.create_model(lname, pretrained=False, num_classes=1)
        except Exception as e:
            raise ValueError(f"Unknown model: {model_name}. Error: {e}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint saved with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded checkpoint with metadata (trained for {checkpoint.get('train_time', 'unknown')} seconds)")
    else:
        # Checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("  ✓ Model loaded successfully")
    
    return model, model_name


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate model on test set.
    
    Returns:
        Tuple of (y_true, y_pred, y_probs, inference_time)
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Mixed precision inference (only supported on CUDA)
            # MPS and CPU don't support torch.amp.autocast well, so disable for those
            use_amp_inference = use_amp and device == 'cuda'
            
            if use_amp_inference:
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(images).squeeze(1)
                    probs = torch.sigmoid(logits)
            else:
                logits = model(images).squeeze(1)
                probs = torch.sigmoid(logits)
            
            # Binary classification: threshold at 0.5
            predicted = (probs > 0.5).long()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    inference_time = time.time() - start_time
    
    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_probs),
        inference_time
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    model_name: str,
    model_path: str,
    inference_time: float,
    num_samples: int,
) -> TestMetrics:
    """Compute comprehensive evaluation metrics."""
    
    # Validate and clean y_probs
    if np.any(np.isnan(y_probs)) or np.any(np.isinf(y_probs)):
        print("  Warning: Invalid probabilities detected, clipping to [0, 1]")
        y_probs = np.nan_to_num(y_probs, nan=0.5, posinf=1.0, neginf=0.0)
        y_probs = np.clip(y_probs, 0, 1)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Compute confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=['Healthy', 'Diseased'],
        digits=4
    )
    
    return TestMetrics(
        model_name=model_name,
        model_path=model_path,
        inference_time=inference_time,
        samples_per_second=num_samples / inference_time if inference_time > 0 else 0,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1_score=f1_score(y_true, y_pred, zero_division=0),
        auc_roc=roc_auc_score(y_true, y_probs),
        confusion_matrix=cm.tolist(),
        classification_report=class_report,
        roc_fpr=fpr.tolist(),
        roc_tpr=tpr.tolist(),
        roc_thresholds=thresholds.tolist(),
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
    )


# ============================================================================
# Testing Pipeline
# ============================================================================

def test_all_models(
    models_dir: str,
    dataset_root: str,
    config: TestConfig,
    output_dir: str = "data/output"
) -> List[TestMetrics]:
    """
    Test all trained models on PlantDoc dataset.
    
    Args:
        models_dir: Directory containing trained model checkpoints
        dataset_root: Root directory of PlantDoc dataset
        config: Test configuration
        output_dir: Directory to save results
    
    Returns:
        List of TestMetrics for each model
    """
    print("\n" + "="*70)
    print("TESTING TRAINED MODELS ON PLANTDOC DATASET")
    print("="*70)
    print(f"Device: {config.device.upper()}")
    if config.device == "mps":
        print("  (Apple Silicon GPU - Metal Performance Shaders)")
    elif config.device == "cuda":
        print(f"  (NVIDIA GPU: {torch.cuda.get_device_name(0)})")
    else:
        print("  (CPU)")
    print(f"Batch Size: {config.batch_size}")
    print(f"Mixed Precision: {config.use_amp} (only active on CUDA)")
    
    # Load PlantDoc test set
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = PlantDocBinaryDataset(
        root_dir=dataset_root,
        split="test",
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Find all model checkpoints
    model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
    
    if not model_paths:
        print(f"\nError: No model checkpoints found in {models_dir}")
        return []
    
    print(f"\nFound {len(model_paths)} model checkpoint(s):")
    for path in model_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Test each model
    results = []
    for model_path in model_paths:
        try:
            print(f"\n{'#'*70}")
            print(f"# Testing {os.path.basename(model_path)}")
            print(f"{'#'*70}")
            
            # Load model
            model, model_name = load_trained_model(model_path, config.device)
            
            # Evaluate
            print("Running inference on test set...")
            y_true, y_pred, y_probs, inference_time = evaluate_model(
                model, test_loader, config.device, use_amp=config.use_amp
            )
            
            # Compute metrics
            metrics = compute_metrics(
                y_true, y_pred, y_probs,
                model_name, model_path,
                inference_time, len(test_dataset)
            )
            
            # Print results
            print(f"\n{model_name} Results:")
            print(f"  Accuracy:  {metrics.accuracy:.4f}")
            print(f"  Precision: {metrics.precision:.4f}")
            print(f"  Recall:    {metrics.recall:.4f}")
            print(f"  F1-Score:  {metrics.f1_score:.4f}")
            print(f"  AUC-ROC:   {metrics.auc_roc:.4f}")
            print(f"  Inference Time: {metrics.inference_time:.2f}s")
            print(f"  Samples/sec:    {metrics.samples_per_second:.1f}")
            print("\nConfusion Matrix:")
            print(f"  TN: {metrics.true_negatives:4d}  FP: {metrics.false_positives:4d}")
            print(f"  FN: {metrics.false_negatives:4d}  TP: {metrics.true_positives:4d}")
            
            results.append(metrics)
            
            # Cleanup
            del model
            if config.device == "cuda":
                torch.cuda.empty_cache()
            elif config.device == "mps":
                torch.mps.empty_cache()
        
        except Exception as e:
            print(f"\nError testing {os.path.basename(model_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_metrics_comparison(results: List[TestMetrics], save_path: str = None):
    """Plot comparison of key metrics across models."""
    if not results:
        return
    
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
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_to_plot)))
    
    for (metric_name, values), color in zip(metrics_to_plot.items(), colors):
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric_name, color=color)
        ax.bar_label(rects, fmt='%.3f', padding=3, fontsize=8)
        multiplier += 1
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('PlantDoc Test Set - Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nMetrics comparison saved to {save_path}")
    
    plt.show()


def plot_confusion_matrices(results: List[TestMetrics], save_path: str = None):
    """Plot confusion matrices for all models."""
    if not results:
        return
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        cm = np.array(result.confusion_matrix)
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Healthy', 'Diseased'],
            yticklabels=['Healthy', 'Diseased'],
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_title(f'{result.model_name}\nAccuracy: {result.accuracy:.4f}', 
                    fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Add statistics
        tn, fp, fn, tp = cm.ravel()
        stats_text = f'Sensitivity: {tp/(tp+fn):.3f}\nSpecificity: {tn/(tn+fp):.3f}'
        ax.text(1.5, -0.15, stats_text, transform=ax.transData, 
               fontsize=9, verticalalignment='top')
    
    plt.suptitle('Confusion Matrices - PlantDoc Test Set', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    
    plt.show()


def plot_roc_curves(results: List[TestMetrics], save_path: str = None):
    """Plot ROC curves for all models."""
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for result, color in zip(results, colors):
        ax.plot(
            result.roc_fpr, result.roc_tpr,
            label=f'{result.model_name} (AUC = {result.auc_roc:.4f})',
            color=color, linewidth=2
        )
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - PlantDoc Test Set', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def plot_inference_speed(results: List[TestMetrics], save_path: str = None):
    """Plot inference speed comparison."""
    if not results:
        return
    
    model_names = [r.model_name for r in results]
    samples_per_sec = [r.samples_per_second for r in results]
    inference_times = [r.inference_time for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Samples per second
    bars1 = ax1.bar(model_names, samples_per_sec, color=plt.cm.viridis(np.linspace(0, 1, len(results))))
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Samples per Second', fontsize=12)
    ax1.set_title('Inference Speed', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, samples_per_sec):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Total inference time
    bars2 = ax2.bar(model_names, inference_times, color=plt.cm.plasma(np.linspace(0, 1, len(results))))
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Total Time (seconds)', fontsize=12)
    ax2.set_title('Total Inference Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, inference_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inference speed comparison saved to {save_path}")
    
    plt.show()


def save_results(results: List[TestMetrics], output_dir: str = "data/output"):
    """Save test results to JSON and create visualizations."""
    if not results:
        print("No results to save")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"plantdoc_test_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    plot_metrics_comparison(
        results,
        save_path=os.path.join(output_dir, f"plantdoc_metrics_comparison_{timestamp}.png")
    )
    
    plot_confusion_matrices(
        results,
        save_path=os.path.join(output_dir, f"plantdoc_confusion_matrices_{timestamp}.png")
    )
    
    plot_roc_curves(
        results,
        save_path=os.path.join(output_dir, f"plantdoc_roc_curves_{timestamp}.png")
    )
    
    plot_inference_speed(
        results,
        save_path=os.path.join(output_dir, f"plantdoc_inference_speed_{timestamp}.png")
    )
    
    # Create summary table
    print(f"\n{'='*90}")
    print("PLANTDOC TEST SET - COMPREHENSIVE SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'Infer(s)':<10} {'Samp/s':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r.model_name:<20} {r.accuracy:<8.4f} {r.precision:<8.4f} "
              f"{r.recall:<8.4f} {r.f1_score:<8.4f} {r.auc_roc:<8.4f} "
              f"{r.inference_time:<10.2f} {r.samples_per_second:<10.1f}")
    print("=" * 90)
    
    # Print detailed classification reports
    print(f"\n{'='*90}")
    print("DETAILED CLASSIFICATION REPORTS")
    print(f"{'='*90}")
    for r in results:
        print(f"\n{r.model_name}:")
        print("-" * 70)
        print(r.classification_report)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main testing pipeline."""
    
    # Configuration
    config = TestConfig(
        batch_size=32,
        num_workers=4,
        use_amp=True,
    )
    
    # Paths - Updated for current workspace structure
    workspace_root = Path(__file__).parent.parent  # Go up from notebooks/ to project root
    models_dir = str(workspace_root / "notebooks" / "models")
    dataset_root = str(workspace_root / "data" / "plantdoc")
    output_dir = str(workspace_root / "data" / "output")
    
    print("\nUsing paths:")
    print(f"  Models: {models_dir}")
    print(f"  Dataset: {dataset_root}")
    print(f"  Output: {output_dir}")
    
    # Check if paths exist
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found: {models_dir}")
        return
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset not found: {dataset_root}")
        return
    
    # Run tests
    results = test_all_models(
        models_dir=models_dir,
        dataset_root=dataset_root,
        config=config,
        output_dir=output_dir
    )
    
    # Save and visualize results
    if results:
        save_results(results, output_dir=output_dir)
        print(f"\n{'='*90}")
        print("✓ Testing complete! All results saved.")
        print(f"{'='*90}")
    else:
        print("\nNo models were successfully tested.")


if __name__ == "__main__":
    main()
