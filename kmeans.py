import os
import torch
import time
import wandb
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils import (
    DEVICE, BATCH_SIZE,
    create_dataloader,
    WandbLogger,
    save_metrics,
    calculate_metrics,
    log_epoch_metrics
)

def extract_simple_features(data_loader):
    """Extract enhanced statistical features from images"""
    features = []
    labels = []
    
    for data, target in tqdm(data_loader, desc="Extracting features"):
        batch_data = data.numpy()
        batch_size = batch_data.shape[0]
        
        batch_features = []
        for i in range(batch_size):
            img = batch_data[i].reshape(1, 150, 150)  # Keep 2D structure
            
            # Basic statistical features
            basic_stats = [
                np.mean(img),
                np.std(img),
                np.median(img),
                np.max(img),
                np.min(img),
            ]
            
            # Histogram features (10 bins)
            hist, _ = np.histogram(img, bins=10, range=(-1, 1))
            hist = hist / hist.sum()  # Normalize
            
            # Edge features using gradient
            dx = np.gradient(img[0], axis=1)
            dy = np.gradient(img[0], axis=0)
            gradient_mag = np.sqrt(dx**2 + dy**2)
            edge_features = [
                np.mean(gradient_mag),
                np.std(gradient_mag),
                np.percentile(gradient_mag, 90),  # Strong edges
            ]
            
            # Region features
            regions = [
                np.mean(img[0, :75, :75]),    # Top-left
                np.mean(img[0, :75, 75:]),    # Top-right
                np.mean(img[0, 75:, :75]),    # Bottom-left
                np.mean(img[0, 75:, 75:]),    # Bottom-right
            ]
            
            # Combine all features
            features_vec = np.concatenate([
                basic_stats,
                hist,
                edge_features,
                regions
            ])
            
            batch_features.append(features_vec)
            
        features.extend(batch_features)
        labels.extend(target.numpy())
    
    return np.array(features), np.array(labels)

def main():
    try:
        # Initialize wandb logger
        config = {
            'batch_size': BATCH_SIZE,
            'feature_extractor': 'simple_statistics',
            'n_clusters': 16,  # Increased number of clusters
            'device': str(DEVICE),
            'model_architecture': 'KMeans'
        }
        wandb_logger = WandbLogger("KMeans_CatDog", config)
        
        # Load data
        train_loader = create_dataloader('train', BATCH_SIZE)
        test_loader = create_dataloader('test', BATCH_SIZE)
        
        # Extract features
        start_time = time.time()
        X_train, Y_train = extract_simple_features(train_loader)
        X_test, Y_test = extract_simple_features(test_loader)
        feature_time = time.time() - start_time
        
        # Train KMeans with improved parameters
        train_start = time.time()
        kmeans = KMeans(
            n_clusters=16,          # More clusters for better separation
            n_init=10,              # Multiple initializations
            max_iter=300,           # More iterations for convergence
            random_state=42,
            init='k-means++'        # Better initialization method
        )
        kmeans.fit(X_train)
        training_time = time.time() - train_start
        
        # Evaluate with improved cluster mapping
        eval_start = time.time()
        Y_pred_clusters = kmeans.predict(X_test)
        
        # Map clusters to labels using training data distribution
        cluster_mapping = {}
        for cluster in np.unique(Y_pred_clusters):
            mask = kmeans.labels_ == cluster
            if np.sum(mask) > 0:
                # Get most common label in cluster
                most_common_label = mode(Y_train[mask], keepdims=True).mode[0]
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
                'gpu_memory_used': 0,  # KMeans runs on CPU
                'gpu_utilization': 0
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