import os
import torch
import torch.nn as nn
import snntorch as snn
from tqdm import tqdm
import time
import wandb
from torch.cuda.amp import autocast, GradScaler
from utils import (
    DEVICE, BATCH_SIZE,
    create_dataloader,
    evaluate_model,
    WandbLogger,
    save_metrics,
    log_epoch_metrics
)

# Model hyperparameters
TIME_STEPS = 8           
LEARNING_RATE = 0.001    
EPOCHS = 30              
DROPOUT_RATE = 0.3

class SNN_Model(nn.Module):
    """Spiking Neural Network for image classification
    
    Architecture:
    - Conv layers: Feature extraction
    - BatchNorm: Training stability
    - MaxPool: Downsampling
    - LIF neurons: Spike encoding
    - FC layers: Classification
    """
    def __init__(self):
        super().__init__()
        # Convolutional layers: Extract features, increase channels
        self.conv1 = nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1)      # 64->96
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)    # 128->192
        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)   # 128->256
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)   # Add an extra layer

        # Batch normalization layers: Accelerate training convergence
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)

        # Other layers
        self.pool = nn.MaxPool2d(2)                    # Max pooling: Dimensionality reduction
        self.dropout = nn.Dropout(DROPOUT_RATE)        # Dropout: Prevent overfitting
        
        # Two LIF neurons
        self.lif1 = snn.Leaky(beta=0.95, threshold=0.5)  # Increase beta value
        self.lif2 = snn.Leaky(beta=0.95, threshold=0.5)
        
        # Use more efficient activation function
        self.act = nn.ReLU(inplace=True)  # Inplace operation saves memory
        
        # Dynamically calculate fully connected layer input dimension
        self._to_linear = None
        self._find_size()
        
        # Fully connected layers: Final classification
        self.fc1 = nn.Linear(self._to_linear, 512)    # 256->512
        self.fc2 = nn.Linear(512, 256)                # Add an extra layer
        self.fc3 = nn.Linear(256, 2)                  # Output layer
    
    def _find_size(self):
        """Calculate FC layer input dimension via forward pass"""
        x = torch.randn(1, 1, 150, 150)
        # First convolutional block
        x = self.act(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)

        # Second convolutional block
        x = self.act(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)

        # Third convolutional block
        x = self.act(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)

        x = self.act(self.conv4(x))
        x = self.bn4(x)
        x = self.pool(x)

        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
    
    def forward(self, x):
        """Forward pass through the network
        
        Args:
            x: Input image [batch_size, 1, 150, 150]
        Returns:
            out: Classification logits [batch_size, 2]
        """
        # Reduce redundant computations within time steps
        # First perform convolutional feature extraction
        x = self.act(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        
        x = self.act(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        
        x = self.act(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)
        
        x = self.act(self.conv4(x))
        x = self.bn4(x)
        x = self.pool(x)
        
        # Then iterate over time steps on the feature map
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []
        
        for _ in range(TIME_STEPS):
            spk1, mem1 = self.lif1(x, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            spk_rec.append(spk2)
        
        # Calculate the average response over the time dimension
        out = torch.stack(spk_rec).mean(dim=0)
        out = out.view(out.size(0), -1)
        
        # Fully connected classification layers
        out = self.dropout(out)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return out

def train_model(model, train_loader, criterion, optimizer, wandb_logger, current_epoch):
    """Train model for one epoch and log metrics"""
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

def main():
    """Main training loop:
    1. Setup wandb logging
    2. Initialize model and data
    3. Train for specified epochs
    4. Save best model and metrics
    """
    try:
        # Initialize wandb logger
        config = {
            'batch_size': BATCH_SIZE,
            'time_steps': TIME_STEPS,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'dropout_rate': DROPOUT_RATE,
            'device': str(DEVICE),
            'model_architecture': 'SNN_Model'
        }
        wandb_logger = WandbLogger("SNN_CatDog", config)
        
        # Load data and initialize model
        train_loader = create_dataloader('train', BATCH_SIZE)
        test_loader = create_dataloader('test', BATCH_SIZE)
        model = SNN_Model().to(DEVICE)
        
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
