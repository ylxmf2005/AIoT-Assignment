import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import time
import wandb
import numpy as np
from scipy.stats import mode

from utils import (
    DEVICE, BATCH_SIZE,
    create_dataloader,
    WandbLogger,
    save_metrics,
    calculate_metrics,
    log_epoch_metrics
)

def extract_features(model, data_loader):
    """Extract features using pretrained VGG16 model"""
    features = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Extracting features"):
            if data.size(1) == 1:
                data = data.repeat(1, 3, 1, 1)
            
            data = data.to(DEVICE)
            # Get features from the last convolutional layer
            feature = model.features(data)
            # Global average pooling
            feature = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1))
            feature = feature.view(feature.size(0), -1)
            features.append(feature.cpu().numpy())
            labels.extend(target.numpy())
    
    return np.vstack(features), np.array(labels)

def main():
    try:
        # Initialize wandb logger
        config = {
            'batch_size': BATCH_SIZE,
            'feature_extractor': 'VGG16',
            'n_clusters': 8,
            'device': str(DEVICE),
            'model_architecture': 'KMeans'
        }
        wandb_logger = WandbLogger("KMeans_CatDog", config)
        
        # Load data
        train_loader = create_dataloader('train', BATCH_SIZE)
        test_loader = create_dataloader('test', BATCH_SIZE)
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(weights='IMAGENET1K_V1').to(DEVICE)
        
        # Extract features
        start_time = time.time()
        X_train, Y_train = extract_features(vgg16, train_loader)
        X_test, Y_test = extract_features(vgg16, test_loader)
        feature_time = time.time() - start_time
        
        # Train KMeans
        train_start = time.time()
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=8, random_state=42)
        kmeans.fit(X_train)
        training_time = time.time() - train_start
        
        # Evaluate
        eval_start = time.time()
        Y_pred_clusters = kmeans.predict(X_test)
        
        # Map clusters to labels
        cluster_mapping = {}
        for cluster in np.unique(Y_pred_clusters):
            most_common_label = mode(Y_test[Y_pred_clusters == cluster], keepdims=True).mode[0]
            cluster_mapping[cluster] = most_common_label
        
        Y_pred = np.array([cluster_mapping[c] for c in Y_pred_clusters])
        eval_time = time.time() - eval_start
        
        # Calculate metrics
        metrics = calculate_metrics(
            Y_test, Y_pred,
            runtime_stats={
                'feature_extraction_time': feature_time,
                'training_time': training_time,
                'evaluation_time': eval_time,
                'total_time': time.time() - start_time,
                'avg_batch_time': feature_time / len(train_loader),
                'avg_inference_time': eval_time / len(test_loader),
                'throughput': len(Y_test) / eval_time,
                'gpu_memory_used': torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
                'gpu_utilization': 100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            }
        )
        
        # Log metrics
        log_epoch_metrics(wandb_logger, metrics)
        
        # Log final metrics
        wandb_logger.log_metrics({
            'final/accuracy': metrics['accuracy'],
            'final/macro_f1': metrics['macro_f1'],
            'final/macro_precision': metrics['macro_precision'],
            'final/macro_recall': metrics['macro_recall'],
            'final/total_time': metrics['runtime_metrics']['total_time']
        })
        
        # Save metrics to file
        save_metrics(metrics, wandb.run.dir)
        
    except Exception as e:
        wandb.finish(exit_code=1)
        raise e
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 