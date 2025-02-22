import os
import torch
import json
import wandb
import numpy as np
from datetime import datetime
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Any, Optional
import time

# Global constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

class CatDogDataset(Dataset):
    """Custom dataset class for handling HuggingFace datasets"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloader(split='train', batch_size=32):
    """Create data loader with augmentation"""
    dataset = load_dataset("Bingsu/Cat_and_Dog")[split]
    
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        *([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0, shear=15,
                scale=(0.8, 1.3),
                translate=(0.1, 0.1)
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3
            ),
            transforms.RandomPerspective(0.2),
        ] if split == 'train' else []),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    custom_dataset = CatDogDataset(dataset, transform=transform)
    
    return DataLoader(
        custom_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

class WandbLogger:
    """Weights & Biases logger for experiment tracking"""
    def __init__(self, project_name: str, config: Dict[str, Any]):
        self.run = wandb.init(
            project=project_name,
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        self.step = 0
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb"""
        if step is not None:
            self.step = step
        wandb.log(metrics, step=self.step)
        self.step += 1

    def finish(self):
        """End wandb logging session"""
        wandb.finish()

def save_metrics(metrics: Dict[str, Any], run_dir: str):
    """Save training metrics to JSON file"""
    def convert_to_json_safe(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj

    json_safe_metrics = {
        k: convert_to_json_safe(v) if isinstance(v, (np.ndarray, torch.Tensor)) 
        else v for k, v in metrics.items()
    }
    
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(json_safe_metrics, f, indent=4)

def calculate_metrics(all_targets, all_preds, loss=0.0, runtime_stats=None):
    """Calculate detailed classification metrics
    
    Args:
        all_targets: Ground truth labels
        all_preds: Model predictions
        loss: Optional loss value
        runtime_stats: Optional dict with runtime statistics
    
    Returns:
        Dictionary containing all metrics
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Per-class metrics
    tp_cat = cm[0][0]  # True positives for cat class
    fp_cat = cm[1][0]  # False positives for cat class
    fn_cat = cm[0][1]  # False negatives for cat class
    tn_cat = cm[1][1]  # True negatives for cat class
    
    tp_dog = cm[1][1]  # True positives for dog class
    fp_dog = cm[0][1]  # False positives for dog class
    fn_dog = cm[1][0]  # False negatives for dog class
    tn_dog = cm[0][0]  # True negatives for dog class
    
    # Calculate rates
    precision_cat = tp_cat / (tp_cat + fp_cat) * 100 if (tp_cat + fp_cat) > 0 else 0
    recall_cat = tp_cat / (tp_cat + fn_cat) * 100 if (tp_cat + fn_cat) > 0 else 0
    f1_cat = 2 * (precision_cat * recall_cat) / (precision_cat + recall_cat) if (precision_cat + recall_cat) > 0 else 0
    
    precision_dog = tp_dog / (tp_dog + fp_dog) * 100 if (tp_dog + fp_dog) > 0 else 0
    recall_dog = tp_dog / (tp_dog + fn_dog) * 100 if (tp_dog + fn_dog) > 0 else 0
    f1_dog = 2 * (precision_dog * recall_dog) / (precision_dog + recall_dog) if (precision_dog + recall_dog) > 0 else 0
    
    # Macro/micro averages
    macro_precision = (precision_cat + precision_dog) / 2
    macro_recall = (recall_cat + recall_dog) / 2
    macro_f1 = (f1_cat + f1_dog) / 2
    
    total = len(all_targets)
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    accuracy = 100 * correct / total
    
    metrics = {
        'accuracy': accuracy,
        'loss': loss,
        'confusion_matrix': cm,
        
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        
        'cat_metrics': {
            'precision': precision_cat,
            'recall': recall_cat,
            'f1': f1_cat,
            'tp': int(tp_cat),
            'fp': int(fp_cat),
            'fn': int(fn_cat),
            'tn': int(tn_cat)
        },
        
        'dog_metrics': {
            'precision': precision_dog,
            'recall': recall_dog,
            'f1': f1_dog,
            'tp': int(tp_dog),
            'fp': int(fp_dog),
            'fn': int(fn_dog),
            'tn': int(tn_dog)
        }
    }
    
    if runtime_stats:
        metrics['runtime_metrics'] = runtime_stats
        
    return metrics

def log_epoch_metrics(wandb_logger, metrics, train_loss=None, train_acc=None, lr=None):
    """Log standardized epoch metrics to wandb
    
    Args:
        wandb_logger: WandbLogger instance
        metrics: Dictionary of evaluation metrics
        train_loss: Optional training loss
        train_acc: Optional training accuracy
        lr: Optional learning rate
    """
    epoch_metrics = {}
    
    if train_loss is not None:
        epoch_metrics['epoch/train_loss'] = train_loss
    if train_acc is not None:
        epoch_metrics['epoch/train_accuracy'] = train_acc
    
    # Add evaluation metrics
    epoch_metrics.update({
        'epoch/val_loss': metrics['loss'],
        'epoch/val_accuracy': metrics['accuracy'],
        
        'epoch/macro_f1': metrics['macro_f1'],
        'epoch/macro_precision': metrics['macro_precision'],
        'epoch/macro_recall': metrics['macro_recall'],
        
        'epoch/cat/precision': metrics['cat_metrics']['precision'],
        'epoch/cat/recall': metrics['cat_metrics']['recall'],
        'epoch/cat/f1': metrics['cat_metrics']['f1'],
        'epoch/cat/tp': metrics['cat_metrics']['tp'],
        'epoch/cat/fp': metrics['cat_metrics']['fp'],
        'epoch/cat/fn': metrics['cat_metrics']['fn'],
        'epoch/cat/tn': metrics['cat_metrics']['tn'],
        
        'epoch/dog/precision': metrics['dog_metrics']['precision'],
        'epoch/dog/recall': metrics['dog_metrics']['recall'],
        'epoch/dog/f1': metrics['dog_metrics']['f1'],
        'epoch/dog/tp': metrics['dog_metrics']['tp'],
        'epoch/dog/fp': metrics['dog_metrics']['fp'],
        'epoch/dog/fn': metrics['dog_metrics']['fn'],
        'epoch/dog/tn': metrics['dog_metrics']['tn'],
    })
    
    if 'runtime_metrics' in metrics:
        epoch_metrics.update({
            'epoch/runtime/batch_time': metrics['runtime_metrics']['avg_batch_time'],
            'epoch/runtime/inference_time': metrics['runtime_metrics']['avg_inference_time'],
            'epoch/runtime/throughput': metrics['runtime_metrics']['throughput'],
            'epoch/runtime/gpu_memory': metrics['runtime_metrics']['gpu_memory_used'],
            'epoch/runtime/gpu_utilization': metrics['runtime_metrics']['gpu_utilization'],
        })
    
    if lr is not None:
        epoch_metrics['epoch/learning_rate'] = lr
        
    wandb_logger.log_metrics(epoch_metrics)

def evaluate_model(model, test_loader, criterion, return_metrics=False):
    """Generic model evaluation function"""
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0
    
    batch_times = []
    inference_times = []
    total_start = time.time()
    
    with torch.no_grad():
        for data, targets in test_loader:
            batch_start = time.time()
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            inference_start = time.time()
            outputs = model(data)
            inference_time = time.time() - inference_start
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            batch_times.append(time.time() - batch_start)
            inference_times.append(inference_time)
    
    avg_loss = test_loss / len(test_loader)
    total_time = time.time() - total_start
    
    runtime_stats = {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times),
        'avg_inference_time': np.mean(inference_times),
        'throughput': len(all_targets) / total_time,
        'gpu_memory_used': torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
        'gpu_utilization': 100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    }
    
    metrics = calculate_metrics(all_targets, all_preds, avg_loss, runtime_stats)
    return metrics if return_metrics else metrics['accuracy'] 