"""
Vision LLM Evaluation Module

Comprehensive evaluation utilities for Vision LLMs:
- VQA accuracy metrics
- Image captioning metrics (BLEU, CIDEr, METEOR)
- Grounding accuracy
- OCR evaluation
- Multi-task benchmark support
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    batch_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.0  # Greedy decoding for eval
    num_beams: int = 1
    do_sample: bool = False
    
    # Metrics to compute
    compute_vqa_accuracy: bool = True
    compute_bleu: bool = True
    compute_cider: bool = False  # Requires pycocoevalcap
    compute_meteor: bool = False
    
    # Output
    output_dir: str = "./eval_results"
    save_predictions: bool = True


class VQAAccuracyMetric:
    """
    VQA accuracy metric following VQAv2 evaluation protocol.
    
    For each answer, accuracy = min(#humans that provided that answer / 3, 1)
    """
    
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.results = []
    
    def reset(self):
        self.correct = 0
        self.total = 0
        self.results = []
    
    def update(
        self,
        predictions: List[str],
        ground_truths: List[List[str]],
    ):
        """
        Update metrics with batch of predictions.
        
        Args:
            predictions: Model predictions
            ground_truths: List of acceptable answers for each question
        """
        for pred, gts in zip(predictions, ground_truths):
            pred_clean = self._clean_answer(pred)
            
            # Count matching ground truths
            matches = sum(1 for gt in gts if self._clean_answer(gt) == pred_clean)
            accuracy = min(matches / 3.0, 1.0)
            
            self.correct += accuracy
            self.total += 1
            
            self.results.append({
                "prediction": pred,
                "ground_truths": gts,
                "accuracy": accuracy,
            })
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and normalize answer string."""
        answer = answer.lower().strip()
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        # Normalize whitespace
        answer = ' '.join(answer.split())
        return answer
    
    def compute(self) -> Dict[str, float]:
        """Compute final accuracy."""
        if self.total == 0:
            return {"vqa_accuracy": 0.0}
        return {"vqa_accuracy": self.correct / self.total * 100}


class BLEUMetric:
    """
    BLEU score for image captioning evaluation.
    """
    
    def __init__(self, n_gram: int = 4):
        self.n_gram = n_gram
        self.predictions = []
        self.references = []
    
    def reset(self):
        self.predictions = []
        self.references = []
    
    def update(
        self,
        predictions: List[str],
        references: List[List[str]],
    ):
        """Update with batch of predictions and references."""
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def compute(self) -> Dict[str, float]:
        """Compute BLEU scores."""
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        except ImportError:
            logger.warning("NLTK not installed. Install with: pip install nltk")
            return {}
        
        # Tokenize
        pred_tokens = [p.lower().split() for p in self.predictions]
        ref_tokens = [[r.lower().split() for r in refs] for refs in self.references]
        
        smoothie = SmoothingFunction().method1
        
        results = {}
        for n in range(1, self.n_gram + 1):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            score = corpus_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothie)
            results[f"bleu_{n}"] = score * 100
        
        return results


class GroundingAccuracyMetric:
    """
    Grounding accuracy for visual grounding tasks.
    
    Measures IoU between predicted and ground truth bounding boxes.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.correct = 0
        self.total = 0
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(
        self,
        pred_boxes: List[Tuple[float, float, float, float]],
        gt_boxes: List[Tuple[float, float, float, float]],
    ):
        """
        Update with predicted and ground truth boxes.
        
        Boxes are in format (x1, y1, x2, y2) normalized to [0, 1].
        """
        for pred, gt in zip(pred_boxes, gt_boxes):
            iou = self._compute_iou(pred, gt)
            if iou >= self.iou_threshold:
                self.correct += 1
            self.total += 1
    
    def _compute_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def compute(self) -> Dict[str, float]:
        """Compute grounding accuracy."""
        if self.total == 0:
            return {"grounding_accuracy": 0.0}
        return {"grounding_accuracy": self.correct / self.total * 100}


class OCRAccuracyMetric:
    """
    OCR accuracy metric using Character Error Rate (CER) and Word Error Rate (WER).
    """
    
    def __init__(self):
        self.predictions = []
        self.references = []
    
    def reset(self):
        self.predictions = []
        self.references = []
    
    def update(self, predictions: List[str], references: List[str]):
        self.predictions.extend(predictions)
        self.references.extend(references)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def compute(self) -> Dict[str, float]:
        """Compute CER and WER."""
        total_chars = 0
        total_char_errors = 0
        total_words = 0
        total_word_errors = 0
        
        for pred, ref in zip(self.predictions, self.references):
            # Character Error Rate
            total_chars += len(ref)
            total_char_errors += self._levenshtein_distance(pred, ref)
            
            # Word Error Rate
            pred_words = pred.split()
            ref_words = ref.split()
            total_words += len(ref_words)
            total_word_errors += self._levenshtein_distance(
                " ".join(pred_words), " ".join(ref_words)
            )
        
        cer = total_char_errors / total_chars * 100 if total_chars > 0 else 0
        wer = total_word_errors / total_words * 100 if total_words > 0 else 0
        
        return {
            "cer": cer,
            "wer": wer,
            "ocr_accuracy": 100 - cer,
        }


class VisionLLMEvaluator:
    """
    Comprehensive evaluator for Vision LLMs.
    
    Supports multiple evaluation tasks:
    - VQA (Visual Question Answering)
    - Image Captioning
    - Visual Grounding
    - OCR/Document Understanding
    
    Usage:
        evaluator = VisionLLMEvaluator(model, tokenizer)
        results = evaluator.evaluate(eval_dataset, task="vqa")
        print(results)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        processor=None,
        config: Optional[EvaluationConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or EvaluationConfig()
        
        self.device = next(model.parameters()).device
        
        # Initialize metrics
        self.metrics = {
            "vqa": VQAAccuracyMetric(),
            "captioning": BLEUMetric(),
            "grounding": GroundingAccuracyMetric(),
            "ocr": OCRAccuracyMetric(),
        }
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(
        self,
        dataset,
        task: str = "vqa",
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Evaluation dataset
            task: Task type (vqa, captioning, grounding, ocr)
            custom_prompt: Custom prompt template
        
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        all_predictions = []
        all_references = []
        
        print(f"\n{'='*60}")
        print(f"  📊 Evaluating: {task.upper()}")
        print(f"{'='*60}")
        print(f"  Samples: {len(dataset)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"{'='*60}\n")
        
        for batch_idx, batch in enumerate(dataloader):
            # Generate predictions
            predictions = self._generate(batch, task, custom_prompt)
            
            # Get references
            if task == "vqa":
                references = batch.get("answers", batch.get("answer", []))
                if isinstance(references[0], str):
                    references = [[r] for r in references]
                self.metrics["vqa"].update(predictions, references)
                
            elif task == "captioning":
                references = batch.get("captions", batch.get("caption", []))
                if isinstance(references[0], str):
                    references = [[r] for r in references]
                self.metrics["captioning"].update(predictions, references)
                
            elif task == "grounding":
                # Parse bounding boxes from predictions
                pred_boxes = [self._parse_box(p) for p in predictions]
                gt_boxes = batch.get("boxes", batch.get("bbox", []))
                self.metrics["grounding"].update(pred_boxes, gt_boxes)
                references = gt_boxes
                
            elif task == "ocr":
                references = batch.get("text", batch.get("ocr_text", []))
                self.metrics["ocr"].update(predictions, references)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * self.config.batch_size} samples...")
        
        # Compute final metrics
        results = self.metrics[task].compute()
        
        # Print results
        print(f"\n{'='*60}")
        print(f"  📈 Results")
        print(f"{'='*60}")
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")
        print(f"{'='*60}\n")
        
        # Save predictions
        if self.config.save_predictions:
            self._save_predictions(all_predictions, all_references, task, results)
        
        return results
    
    def _generate(
        self,
        batch: Dict[str, Any],
        task: str,
        custom_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate model predictions for a batch."""
        # Build prompts
        if custom_prompt:
            prompts = [custom_prompt.format(**{k: v[i] if isinstance(v, list) else v 
                      for k, v in batch.items()}) for i in range(len(batch["image"]))]
        else:
            prompts = self._build_prompts(batch, task)
        
        # Prepare inputs
        if self.processor is not None:
            inputs = self.processor(
                text=prompts,
                images=batch["image"],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            inputs = {"input_ids": batch.get("input_ids", None)}
            if inputs["input_ids"] is not None:
                inputs["input_ids"] = inputs["input_ids"].to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
        )
        
        # Decode
        if self.tokenizer is not None:
            predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            predictions = [str(o) for o in outputs]
        
        return predictions
    
    def _build_prompts(self, batch: Dict[str, Any], task: str) -> List[str]:
        """Build prompts for different tasks."""
        prompts = []
        
        batch_size = len(batch["image"])
        
        for i in range(batch_size):
            if task == "vqa":
                question = batch.get("question", [""])[i]
                prompt = f"Question: {question}\nAnswer:"
            elif task == "captioning":
                prompt = "Describe this image in detail:"
            elif task == "grounding":
                target = batch.get("target", ["object"])[i]
                prompt = f"Locate the {target} in this image and provide the bounding box:"
            elif task == "ocr":
                prompt = "Read and transcribe all text in this image:"
            else:
                prompt = "Describe this image:"
            
            prompts.append(prompt)
        
        return prompts
    
    def _parse_box(self, text: str) -> Tuple[float, float, float, float]:
        """Parse bounding box from text output."""
        # Try to find box coordinates in text
        patterns = [
            r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',
            r'\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)',
            r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                coords = [float(x) for x in match.groups()]
                # Normalize if needed (assume 1000x1000 if > 1)
                if max(coords) > 1:
                    coords = [c / 1000 for c in coords]
                return tuple(coords)
        
        return (0.0, 0.0, 0.0, 0.0)
    
    def _save_predictions(
        self,
        predictions: List[str],
        references: List[Any],
        task: str,
        results: Dict[str, float],
    ):
        """Save predictions and results to file."""
        output = {
            "task": task,
            "metrics": results,
            "predictions": [
                {"prediction": p, "reference": r}
                for p, r in zip(predictions, references)
            ],
        }
        
        output_path = self.output_dir / f"{task}_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"  💾 Results saved to {output_path}")


def evaluate_vqa(
    model: nn.Module,
    dataset,
    tokenizer=None,
    **kwargs,
) -> Dict[str, float]:
    """Convenience function for VQA evaluation."""
    evaluator = VisionLLMEvaluator(model, tokenizer, **kwargs)
    return evaluator.evaluate(dataset, task="vqa")


def evaluate_captioning(
    model: nn.Module,
    dataset,
    tokenizer=None,
    **kwargs,
) -> Dict[str, float]:
    """Convenience function for captioning evaluation."""
    evaluator = VisionLLMEvaluator(model, tokenizer, **kwargs)
    return evaluator.evaluate(dataset, task="captioning")
