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
    evaluate_model,
    create_dataloader, 
    DEVICE
)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DROPOUT_RATE = 0.3

class CNN_Model(nn.Module):
    """Standard CNN model
    
    VGG-style CNN architecture with:
    - 5 conv blocks, each with 2 conv layers
    - BatchNorm for faster training
    - MaxPool for dimension reduction
    - 3 fully connected layers
    """
    def __init__(self):
        # First block 1 -> 64
        # Second block 64 -> 128
        # Third block 128 -> 256
        # Fourth block 256 -> 512
        # Fifth block 512 -> 512
        super().__init__()
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), # It's a grayscale image, so the channel is 1, the filter number is 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), # inplace=True means the operation will be done in place, which means the operation will be done on the original tensor, rather than creating a new tensor
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # the image size will be reduced by half
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth conv block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fifth conv block
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        # 512 * 4 * 4 -> 1024 -> 256 -> 2
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), # randomly drop some neurons to prevent overfitting
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(1024, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # use kaiming normal, it can adjust the scale of the weights according to the input size
                # we use fan_out mode, it's more focused on the output channel and suitable for ReLU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # keep the original distribution, study the scale gradually
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # normal distribution, mean=0, std=0.01
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, criterion, optimizer, wandb_logger, current_epoch):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    scaler = GradScaler()
    
    try:
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item()
            
            # Log batch metrics
            batch_metrics = {
                'batch/loss': running_loss / (batch_idx + 1),
                'batch/accuracy': 100. * correct / total,
                'batch/learning_rate': optimizer.param_groups[0]['lr']
            }
            wandb_logger.log_metrics(batch_metrics)
            
            pbar.set_postfix({
                'Loss': f'{batch_metrics["batch/loss"]:.3f}',
                'Acc': f'{batch_metrics["batch/accuracy"]:.2f}%'
            })
            
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        return running_loss/len(train_loader), 100.*correct/total
    finally:
        pbar.close()

def evaluate_model(model, test_loader, criterion, return_metrics=False):
    """Evaluate model performance"""
    model.eval()
    total, correct = 0, 0
    all_preds = []
    all_targets = []
    test_loss = 0
    
    batch_times = []
    inference_times = []
    total_start = time.time()
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            batch_start = time.time()
            
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            inference_start = time.time()
            outputs = model(data)
            inference_time = time.time() - inference_start
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            batch_times.append(time.time() - batch_start)
            inference_times.append(inference_time)
    
    total_time = time.time() - total_start
    
    # Calculate confusion matrix and metrics
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
    
    # Calculate precision, recall, and F1 score
    precision_cat = tp_cat / (tp_cat + fp_cat) * 100 if (tp_cat + fp_cat) > 0 else 0
    recall_cat = tp_cat / (tp_cat + fn_cat) * 100 if (tp_cat + fn_cat) > 0 else 0
    f1_cat = 2 * (precision_cat * recall_cat) / (precision_cat + recall_cat) if (precision_cat + recall_cat) > 0 else 0
    
    precision_dog = tp_dog / (tp_dog + fp_dog) * 100 if (tp_dog + fp_dog) > 0 else 0
    recall_dog = tp_dog / (tp_dog + fn_dog) * 100 if (tp_dog + fn_dog) > 0 else 0
    f1_dog = 2 * (precision_dog * recall_dog) / (precision_dog + recall_dog) if (precision_dog + recall_dog) > 0 else 0
    
    # Calculate macro and micro averages
    macro_precision = (precision_cat + precision_dog) / 2
    macro_recall = (recall_cat + recall_dog) / 2
    macro_f1 = (f1_cat + f1_dog) / 2
    
    # Prepare metrics dictionary with the same structure as SNN
    metrics = {
        'accuracy': 100. * correct / total,
        'loss': test_loss / len(test_loader),
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
        },
        'runtime_metrics': {
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'avg_inference_time': np.mean(inference_times),
            'throughput': len(all_targets) / total_time,
            'gpu_memory_used': torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
            'gpu_utilization': 100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        }
    }
    
    return metrics if return_metrics else metrics['accuracy']

def main():
    try:
        # Initialize wandb logger
        config = {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'dropout_rate': DROPOUT_RATE,
            'device': str(DEVICE),
            'model_architecture': 'CNN_Model'
        }
        wandb_logger = WandbLogger("CNN_CatDog", config)
        
        # Load data and initialize model
        train_loader = create_dataloader('train', BATCH_SIZE)
        test_loader = create_dataloader('test', BATCH_SIZE)
        model = CNN_Model().to(DEVICE)
        
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
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        best_acc = 0.0
        
        for epoch in range(EPOCHS):
            # Train
            train_loss, train_acc = train_model(
                model, train_loader, criterion, optimizer, wandb_logger, epoch
            )
            
            # Validate
            metrics = evaluate_model(
                model, 
                test_loader, 
                criterion, 
                return_metrics=True
            )
            
            # Log epoch metrics
            log_epoch_metrics(
                wandb_logger,
                metrics,
                train_loss=train_loss,
                train_acc=train_acc,
                lr=optimizer.param_groups[0]['lr']
            )
            
            # Save best model
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pth'))
            
            scheduler.step()
        
        # Log final metrics
        final_metrics = evaluate_model(model, test_loader, criterion, return_metrics=True)
        wandb_logger.log_metrics({
            'final/accuracy': final_metrics['accuracy'],
            'final/loss': final_metrics['loss'],
            'final/macro_f1': final_metrics['macro_f1'],
            'final/best_accuracy': best_acc,
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
