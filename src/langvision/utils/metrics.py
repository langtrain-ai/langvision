"""
Comprehensive metrics tracking and evaluation utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import sklearn.metrics as sk_metrics
from dataclasses import dataclass
import warnings


@dataclass
class MetricResult:
    """Container for metric computation results."""
    value: float
    count: int
    sum: float
    
    def __post_init__(self):
        if self.count == 0:
            self.value = 0.0


class MetricsTracker:
    """Advanced metrics tracking with support for various metric types."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)
    
    def update(self, name: str, value: float, count: int = 1):
        """Update a metric with a new value."""
        self.metrics[name].append(value)
        self.counts[name] += count
        self.sums[name] += value * count
    
    def get_average(self, name: str) -> float:
        """Get the average value of a metric."""
        if self.counts[name] == 0:
            return 0.0
        return self.sums[name] / self.counts[name]
    
    def get_averages(self, names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get average values for multiple metrics."""
        if names is None:
            names = list(self.metrics.keys())
        
        return {name: self.get_average(name) for name in names}
    
    def get_latest(self, name: str) -> float:
        """Get the latest value of a metric."""
        if not self.metrics[name]:
            return 0.0
        return self.metrics[name][-1]
    
    def reset(self, names: Optional[List[str]] = None):
        """Reset metrics."""
        if names is None:
            self.metrics.clear()
            self.counts.clear()
            self.sums.clear()
        else:
            for name in names:
                if name in self.metrics:
                    del self.metrics[name]
                if name in self.counts:
                    del self.counts[name]
                if name in self.sums:
                    del self.sums[name]
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary of all metrics."""
        summary = {}
        for name in self.metrics.keys():
            values = self.metrics[name]
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values),
                    'latest': values[-1]
                }
        return summary


class ClassificationMetrics:
    """Comprehensive classification metrics computation."""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy."""
        correct = (predictions.argmax(dim=1) == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def top_k_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Compute top-k accuracy."""
        _, top_k_preds = predictions.topk(k, dim=1)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        return correct.any(dim=1).float().mean().item()
    
    @staticmethod
    def precision_recall_f1(predictions: torch.Tensor, 
                           targets: torch.Tensor, 
                           average: str = 'weighted') -> Dict[str, float]:
        """Compute precision, recall, and F1 score."""
        preds = predictions.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        precision = sk_metrics.precision_score(targets, preds, average=average, zero_division=0)
        recall = sk_metrics.recall_score(targets, preds, average=average, zero_division=0)
        f1 = sk_metrics.f1_score(targets, preds, average=average, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """Compute confusion matrix."""
        preds = predictions.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        return sk_metrics.confusion_matrix(targets, preds)
    
    @staticmethod
    def classification_report(predictions: torch.Tensor, 
                            targets: torch.Tensor,
                            class_names: Optional[List[str]] = None) -> str:
        """Generate classification report."""
        preds = predictions.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        return sk_metrics.classification_report(targets, preds, target_names=class_names)


class ContrastiveMetrics:
    """Metrics for contrastive learning and multimodal models."""
    
    @staticmethod
    def contrastive_accuracy(image_features: torch.Tensor, 
                           text_features: torch.Tensor,
                           temperature: float = 0.07) -> Dict[str, float]:
        """Compute contrastive accuracy (image-to-text and text-to-image)."""
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / temperature
        
        # Ground truth labels (diagonal)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Image-to-text accuracy
        i2t_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        
        # Text-to-image accuracy
        t2i_acc = (logits.T.argmax(dim=1) == labels).float().mean().item()
        
        return {
            'i2t_accuracy': i2t_acc,
            't2i_accuracy': t2i_acc,
            'mean_accuracy': (i2t_acc + t2i_acc) / 2
        }
    
    @staticmethod
    def retrieval_metrics(image_features: torch.Tensor,
                         text_features: torch.Tensor,
                         k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """Compute retrieval metrics (Recall@K)."""
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        similarities = torch.matmul(image_features, text_features.T)
        
        batch_size = similarities.shape[0]
        metrics = {}
        
        # Image-to-text retrieval
        for k in k_values:
            if k <= batch_size:
                # Get top-k indices for each image
                _, top_k_indices = similarities.topk(k, dim=1)
                
                # Check if correct text is in top-k
                correct_indices = torch.arange(batch_size, device=similarities.device).unsqueeze(1)
                hits = (top_k_indices == correct_indices).any(dim=1)
                recall_at_k = hits.float().mean().item()
                
                metrics[f'i2t_recall@{k}'] = recall_at_k
        
        # Text-to-image retrieval
        similarities_t2i = similarities.T
        for k in k_values:
            if k <= batch_size:
                _, top_k_indices = similarities_t2i.topk(k, dim=1)
                correct_indices = torch.arange(batch_size, device=similarities.device).unsqueeze(1)
                hits = (top_k_indices == correct_indices).any(dim=1)
                recall_at_k = hits.float().mean().item()
                
                metrics[f't2i_recall@{k}'] = recall_at_k
        
        return metrics


class PerformanceMetrics:
    """Performance and efficiency metrics."""
    
    @staticmethod
    def model_size(model: torch.nn.Module) -> Dict[str, int]:
        """Calculate model size metrics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """Get GPU memory usage statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'max_cached_mb': torch.cuda.max_memory_reserved() / 1024**2
            }
        else:
            return {
                'allocated_mb': 0.0,
                'cached_mb': 0.0,
                'max_allocated_mb': 0.0,
                'max_cached_mb': 0.0
            }
    
    @staticmethod
    def throughput_metrics(batch_size: int, 
                          processing_time: float,
                          num_samples: Optional[int] = None) -> Dict[str, float]:
        """Calculate throughput metrics."""
        if num_samples is None:
            num_samples = batch_size
        
        return {
            'samples_per_second': num_samples / processing_time if processing_time > 0 else 0,
            'batches_per_second': 1 / processing_time if processing_time > 0 else 0,
            'ms_per_sample': (processing_time * 1000) / num_samples if num_samples > 0 else 0,
            'ms_per_batch': processing_time * 1000
        }


class EvaluationSuite:
    """Comprehensive evaluation suite for vision models."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 class_names: Optional[List[str]] = None):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.metrics_tracker = MetricsTracker()
        self.classification_metrics = ClassificationMetrics()
        self.contrastive_metrics = ContrastiveMetrics()
        self.performance_metrics = PerformanceMetrics()
    
    def evaluate_classification(self, 
                              dataloader: torch.utils.data.DataLoader,
                              return_predictions: bool = False) -> Dict[str, Any]:
        """Comprehensive classification evaluation."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # Forward pass
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = self.model(images)
                end_time.record()
                
                torch.cuda.synchronize()
                batch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                
                # Collect predictions and targets
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
                
                # Update metrics
                batch_size = images.size(0)
                acc = self.classification_metrics.accuracy(outputs, targets)
                top5_acc = self.classification_metrics.top_k_accuracy(outputs, targets, k=5)
                
                self.metrics_tracker.update('accuracy', acc, batch_size)
                self.metrics_tracker.update('top5_accuracy', top5_acc, batch_size)
                
                # Performance metrics
                throughput = self.performance_metrics.throughput_metrics(batch_size, batch_time)
                for metric_name, metric_value in throughput.items():
                    self.metrics_tracker.update(metric_name, metric_value, 1)
        
        # Combine all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute comprehensive metrics
        results = self.metrics_tracker.get_averages()
        
        # Add detailed classification metrics
        detailed_metrics = self.classification_metrics.precision_recall_f1(all_predictions, all_targets)
        results.update(detailed_metrics)
        
        # Add confusion matrix
        results['confusion_matrix'] = self.classification_metrics.confusion_matrix(all_predictions, all_targets)
        
        # Add classification report
        if self.class_names:
            results['classification_report'] = self.classification_metrics.classification_report(
                all_predictions, all_targets, self.class_names
            )
        
        # Add model size metrics
        results.update(self.performance_metrics.model_size(self.model))
        
        # Add memory usage
        results.update(self.performance_metrics.memory_usage())
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['targets'] = all_targets
        
        return results
    
    def evaluate_contrastive(self, 
                           dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Evaluate contrastive/multimodal model."""
        self.model.eval()
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                texts = batch.get('texts', None)
                
                if texts is None:
                    warnings.warn("No text data found for contrastive evaluation")
                    continue
                
                # Forward pass
                outputs = self.model(images, texts, return_features=True)
                
                # Collect features
                if 'vision_proj' in outputs and 'text_proj' in outputs:
                    all_image_features.append(outputs['vision_proj'].cpu())
                    all_text_features.append(outputs['text_proj'].cpu())
        
        if not all_image_features:
            return {'error': 'No valid batches for contrastive evaluation'}
        
        # Combine all features
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        
        # Compute contrastive metrics
        results = {}
        
        # Contrastive accuracy
        contrastive_acc = self.contrastive_metrics.contrastive_accuracy(
            all_image_features, all_text_features
        )
        results.update(contrastive_acc)
        
        # Retrieval metrics
        retrieval_metrics = self.contrastive_metrics.retrieval_metrics(
            all_image_features, all_text_features
        )
        results.update(retrieval_metrics)
        
        return results
    
    def benchmark_inference(self, 
                          dataloader: torch.utils.data.DataLoader,
                          num_warmup: int = 10,
                          num_benchmark: int = 100) -> Dict[str, float]:
        """Benchmark model inference performance."""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_warmup:
                    break
                
                images = batch['images'].to(self.device)
                _ = self.model(images)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_benchmark:
                    break
                
                images = batch['images'].to(self.device)
                batch_size = images.size(0)
                
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = self.model(images)
                end_time.record()
                
                torch.cuda.synchronize()
                batch_time = start_time.elapsed_time(end_time) / 1000.0
                times.append(batch_time / batch_size)  # Time per sample
        
        # Compute statistics
        times = np.array(times)
        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'min_inference_time_ms': np.min(times) * 1000,
            'max_inference_time_ms': np.max(times) * 1000,
            'median_inference_time_ms': np.median(times) * 1000,
            'throughput_fps': 1.0 / np.mean(times)
        }
