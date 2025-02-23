import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import time
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import (
    WandbLogger,
    save_metrics,
    log_epoch_metrics,
    create_dataloader,
    DEVICE
)

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
LATENT_DIM = 512

class Autoencoder(nn.Module):
    """Autoencoder architecture for image compression and reconstruction
    
    Architecture:
    - Encoder: Conv2d + BatchNorm + ReLU + MaxPool blocks
    - Latent: Dense layer (512 dimensions)
    - Decoder: Upsample + Conv2d + BatchNorm + ReLU blocks
    
    Dimensions:
    - Input/Output: (batch_size, 1, 150, 150)
    - Encoder path: 150 -> 75 -> 38 -> 19 -> 9
    - Latent: 20736 -> 512 -> 20736
    - Decoder path: 9 -> 18 -> 37 -> 75 -> 150
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1: 150 -> 75
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 75 -> 38
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 38 -> 19
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 19 -> 9
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Latent space
        self.flatten_dim = 256 * 9 * 9  # 20736
        self.flatten = nn.Flatten()
        self.fc_encoder = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)
        self.unflatten = nn.Unflatten(1, (256, 9, 9))
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Block 1: 9 -> 18
            nn.Upsample(size=(18, 18), mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2: 18 -> 37
            nn.Upsample(size=(37, 37), mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 37 -> 75
            nn.Upsample(size=(75, 75), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 4: 75 -> 150
            nn.Upsample(size=(150, 150), mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """Encode input to latent representation"""
        x = self.encoder(x)
        x = self.flatten(x)
        return self.fc_encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        z = self.fc_decoder(z)
        z = self.unflatten(z)
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        if x.shape[-2:] != (150, 150):
            raise ValueError(f"Expected input shape (B, 1, 150, 150), got {x.shape}")
        
        # Get latent representation
        z = self.encode(x)
        
        # Get reconstruction and classification
        reconstructed = self.decode(z)
        logits = self.classifier(z)
        
        if reconstructed.shape[-2:] != (150, 150):
            raise ValueError(f"Expected output shape (B, 1, 150, 150), got {reconstructed.shape}")
        
        return reconstructed, logits

def train_model(model, train_loader, criterion_recon, criterion_class, optimizer, wandb_logger, current_epoch):
    """Train autoencoder for one epoch"""
    model.train()
    running_loss = 0.0
    running_recon_loss = 0.0
    running_class_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {current_epoch}")
    scaler = GradScaler()
    
    try:
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with autocast():
                # Forward pass
                reconstructed, logits = model(data)
                
                # Calculate losses
                recon_loss = criterion_recon(reconstructed, data)
                class_loss = criterion_class(logits, labels)
                
                # Combined loss (with weighting)
                loss = recon_loss + 0.1 * class_loss
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_class_loss += class_loss.item()
            
            # Log batch metrics
            batch_metrics = {
                'batch/loss': running_loss / (batch_idx + 1),
                'batch/recon_loss': running_recon_loss / (batch_idx + 1),
                'batch/class_loss': running_class_loss / (batch_idx + 1),
                'batch/accuracy': 100 * correct / total,
                'batch/learning_rate': optimizer.param_groups[0]['lr']
            }
            wandb_logger.log_metrics(batch_metrics)
            
            pbar.set_postfix({
                'Loss': f'{batch_metrics["batch/loss"]:.3f}',
                'Acc': f'{batch_metrics["batch/accuracy"]:.1f}%'
            })
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        return {
            'loss': running_loss / len(train_loader),
            'recon_loss': running_recon_loss / len(train_loader),
            'class_loss': running_class_loss / len(train_loader),
            'accuracy': 100 * correct / total
        }
    
    finally:
        pbar.close()

def evaluate_autoencoder(model, test_loader, criterion_recon, criterion_class):
    """Evaluate autoencoder performance"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    batch_times = []
    inference_times = []
    total_start = time.time()
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing"):
            batch_start = time.time()
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            inference_start = time.time()
            reconstructed, logits = model(data)
            inference_time = time.time() - inference_start
            
            # Calculate losses
            recon_loss = criterion_recon(reconstructed, data)
            class_loss = criterion_class(logits, labels)
            loss = recon_loss + 0.1 * class_loss
            
            # Get predictions
            _, predicted = torch.max(logits.data, 1)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            batch_times.append(time.time() - batch_start)
            inference_times.append(inference_time)
    
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate per-class metrics
    tp_cat = cm[0][0]  # True positives for cat class
    fp_cat = cm[1][0]  # False positives for cat class
    fn_cat = cm[0][1]  # False negatives for cat class
    tn_cat = cm[1][1]  # True negatives for cat class
    
    tp_dog = cm[1][1]  # True positives for dog class
    fp_dog = cm[0][1]  # False positives for dog class
    fn_dog = cm[1][0]  # False negatives for dog class
    tn_dog = cm[0][0]  # True negatives for dog class
    
    # Calculate metrics
    total = len(all_targets)
    correct = (cm[0][0] + cm[1][1])
    accuracy = 100.0 * correct / total
    
    precision_cat = 100.0 * tp_cat / (tp_cat + fp_cat) if (tp_cat + fp_cat) > 0 else 0
    recall_cat = 100.0 * tp_cat / (tp_cat + fn_cat) if (tp_cat + fn_cat) > 0 else 0
    f1_cat = 2 * (precision_cat * recall_cat) / (precision_cat + recall_cat) if (precision_cat + recall_cat) > 0 else 0
    
    precision_dog = 100.0 * tp_dog / (tp_dog + fp_dog) if (tp_dog + fp_dog) > 0 else 0
    recall_dog = 100.0 * tp_dog / (tp_dog + fn_dog) if (tp_dog + fn_dog) > 0 else 0
    f1_dog = 2 * (precision_dog * recall_dog) / (precision_dog + recall_dog) if (precision_dog + recall_dog) > 0 else 0
    
    # Calculate macro averages
    macro_precision = (precision_cat + precision_dog) / 2
    macro_recall = (recall_cat + recall_dog) / 2
    macro_f1 = (f1_cat + f1_dog) / 2
    
    avg_loss = total_loss / len(test_loader)
    total_time = time.time() - total_start
    
    # Return metrics matching the format in metrics.json
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'confusion_matrix': cm.tolist(),
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
        },
        'runtime_metrics': {
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'avg_inference_time': np.mean(inference_times),
            'throughput': len(test_loader.dataset) / total_time,
            'gpu_memory_used': torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
            'gpu_utilization': 100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        }
    }

def log_epoch_metrics(wandb_logger, metrics, train_loss=None, lr=None):
    """Log standardized epoch metrics to wandb for autoencoder
    
    Args:
        wandb_logger: WandbLogger instance
        metrics: Dictionary of evaluation metrics
        train_loss: Optional training loss
        lr: Optional learning rate
    """
    epoch_metrics = {}
    
    if train_loss is not None:
        epoch_metrics['epoch/train_loss'] = train_loss
    
    # Add evaluation metrics
    epoch_metrics.update({
        'epoch/val_loss': metrics['loss'],
        'epoch/accuracy': metrics['accuracy'],
        'epoch/macro_precision': metrics['macro_precision'],
        'epoch/macro_recall': metrics['macro_recall'],
        'epoch/macro_f1': metrics['macro_f1'],
        
        # Cat class metrics
        'epoch/cat/precision': metrics['cat_metrics']['precision'],
        'epoch/cat/recall': metrics['cat_metrics']['recall'],
        'epoch/cat/f1': metrics['cat_metrics']['f1'],
        
        # Dog class metrics
        'epoch/dog/precision': metrics['dog_metrics']['precision'],
        'epoch/dog/recall': metrics['dog_metrics']['recall'],
        'epoch/dog/f1': metrics['dog_metrics']['f1'],
        
        # Runtime metrics
        'epoch/runtime/batch_time': metrics['runtime_metrics']['avg_batch_time'],
        'epoch/runtime/inference_time': metrics['runtime_metrics']['avg_inference_time'],
        'epoch/runtime/throughput': metrics['runtime_metrics']['throughput'],
        'epoch/runtime/gpu_memory': metrics['runtime_metrics']['gpu_memory_used'],
        'epoch/runtime/gpu_utilization': metrics['runtime_metrics']['gpu_utilization'],
    })
    
    if lr is not None:
        epoch_metrics['epoch/learning_rate'] = lr
        
    wandb_logger.log_metrics(epoch_metrics)

def main():
    try:
        # Initialize wandb logger
        config = {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'latent_dim': LATENT_DIM,
            'device': str(DEVICE),
            'model_architecture': 'Autoencoder_with_Classifier'
        }
        wandb_logger = WandbLogger("Autoencoder_CatDog", config)
        
        # Load data and initialize model
        train_loader = create_dataloader('train', BATCH_SIZE)
        test_loader = create_dataloader('test', BATCH_SIZE)
        model = Autoencoder(LATENT_DIM).to(DEVICE)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        )
        
        criterion_recon = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()
        
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(EPOCHS):
            # Train
            train_metrics = train_model(
                model, train_loader, criterion_recon, criterion_class, optimizer, wandb_logger, epoch
            )
            
            # Validate
            metrics = evaluate_autoencoder(model, test_loader, criterion_recon, criterion_class)
            
            # Log epoch metrics
            log_epoch_metrics(
                wandb_logger,
                metrics,
                train_loss=train_metrics['loss'],
                lr=optimizer.param_groups[0]['lr']
            )
            
            # Save best model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))
            
            scheduler.step()
        
        # Log final metrics
        final_metrics = evaluate_autoencoder(model, test_loader, criterion_recon, criterion_class)
        wandb_logger.log_metrics({
            'final/loss': final_metrics['loss'],
            'final/best_loss': best_loss,
            'final/total_time': time.time() - start_time
        })
        
        # Save metrics to file
        save_metrics(final_metrics, wandb.run.dir)
        
    except Exception as e:
        wandb.finish(exit_code=1)
        raise e
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
