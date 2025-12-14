"""
Model Export Utilities for Vision LLMs

Export fine-tuned models to various formats:
- ONNX for cross-platform deployment
- TorchScript for production serving
- SavedModel for TensorFlow Serving
- OpenVINO for Intel hardware
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    # Export format
    format: str = "onnx"  # onnx, torchscript, safetensors
    
    # ONNX specific
    opset_version: int = 14
    dynamic_axes: bool = True
    simplify: bool = True
    
    # Optimization
    optimize: bool = True
    quantize: bool = False
    quantize_mode: str = "dynamic"  # dynamic, static
    
    # Input specs
    batch_size: int = 1
    image_size: Tuple[int, int] = (384, 384)
    max_seq_length: int = 512
    
    # Output
    output_dir: str = "./exported_models"


class ModelExporter:
    """
    Export Vision LLMs to various deployment formats.
    
    Features:
    - ONNX export with optimization
    - TorchScript compilation
    - Quantization support
    - Metadata preservation
    
    Usage:
        exporter = ModelExporter(model, tokenizer)
        exporter.export("onnx", "./my_model.onnx")
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        processor=None,
        config: Optional[ExportConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or ExportConfig()
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        format: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export model to specified format.
        
        Args:
            format: Export format (onnx, torchscript, safetensors)
            output_path: Output file/directory path
            **kwargs: Format-specific options
        
        Returns:
            Path to exported model
        """
        format = format or self.config.format
        
        if format == "onnx":
            return self.export_onnx(output_path, **kwargs)
        elif format == "torchscript":
            return self.export_torchscript(output_path, **kwargs)
        elif format == "safetensors":
            return self.export_safetensors(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_onnx(
        self,
        output_path: Optional[str] = None,
        opset_version: Optional[int] = None,
        dynamic_axes: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output file path
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic axes for variable batch/sequence
        
        Returns:
            Path to exported ONNX model
        """
        output_path = output_path or str(self.output_dir / "model.onnx")
        opset_version = opset_version or self.config.opset_version
        dynamic_axes = dynamic_axes if dynamic_axes is not None else self.config.dynamic_axes
        
        print(f"\n{'='*60}")
        print(f"  📦 Exporting to ONNX")
        print(f"{'='*60}")
        print(f"  Output: {output_path}")
        print(f"  Opset:  {opset_version}")
        print(f"{'='*60}\n")
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs()
        
        # Dynamic axes configuration
        if dynamic_axes:
            dynamic_axes_dict = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "pixel_values": {0: "batch"},
                "output": {0: "batch", 1: "sequence"},
            }
        else:
            dynamic_axes_dict = None
        
        # Export
        try:
            torch.onnx.export(
                self.model,
                tuple(dummy_inputs.values()),
                output_path,
                input_names=list(dummy_inputs.keys()),
                output_names=["output"],
                opset_version=opset_version,
                dynamic_axes=dynamic_axes_dict,
                do_constant_folding=True,
            )
            
            # Optimize if requested
            if self.config.optimize and self.config.simplify:
                self._simplify_onnx(output_path)
            
            # Quantize if requested
            if self.config.quantize:
                output_path = self._quantize_onnx(output_path)
            
            print(f"  ✅ Export successful: {output_path}")
            
            # Save metadata
            self._save_metadata(output_path, "onnx")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
        
        return output_path
    
    def export_torchscript(
        self,
        output_path: Optional[str] = None,
        method: str = "trace",  # trace or script
        **kwargs,
    ) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            output_path: Output file path
            method: Compilation method (trace or script)
        
        Returns:
            Path to exported TorchScript model
        """
        output_path = output_path or str(self.output_dir / "model.pt")
        
        print(f"\n{'='*60}")
        print(f"  📦 Exporting to TorchScript")
        print(f"{'='*60}")
        print(f"  Output: {output_path}")
        print(f"  Method: {method}")
        print(f"{'='*60}\n")
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs()
        
        try:
            if method == "trace":
                # Trace the model
                traced = torch.jit.trace(
                    self.model,
                    tuple(dummy_inputs.values()),
                )
            else:
                # Script the model
                traced = torch.jit.script(self.model)
            
            # Optimize
            if self.config.optimize:
                traced = torch.jit.optimize_for_inference(traced)
            
            # Save
            traced.save(output_path)
            
            print(f"  ✅ Export successful: {output_path}")
            
            # Save metadata
            self._save_metadata(output_path, "torchscript")
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
        
        return output_path
    
    def export_safetensors(
        self,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export model weights to SafeTensors format.
        
        SafeTensors is a safe, fast format for storing tensors.
        """
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
        
        output_path = output_path or str(self.output_dir / "model.safetensors")
        
        print(f"\n{'='*60}")
        print(f"  📦 Exporting to SafeTensors")
        print(f"{'='*60}")
        print(f"  Output: {output_path}")
        print(f"{'='*60}\n")
        
        # Get state dict
        state_dict = self.model.state_dict()
        
        # Convert to safetensors format (CPU, contiguous)
        tensors = {
            k: v.cpu().contiguous()
            for k, v in state_dict.items()
        }
        
        # Save
        save_file(tensors, output_path)
        
        print(f"  ✅ Export successful: {output_path}")
        
        # Save metadata
        self._save_metadata(output_path, "safetensors")
        
        return output_path
    
    def _create_dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for tracing."""
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        inputs = {}
        
        # Image input
        inputs["pixel_values"] = torch.randn(
            self.config.batch_size,
            3,
            self.config.image_size[0],
            self.config.image_size[1],
            device=device,
            dtype=dtype,
        )
        
        # Text inputs
        inputs["input_ids"] = torch.ones(
            self.config.batch_size,
            self.config.max_seq_length,
            device=device,
            dtype=torch.long,
        )
        
        inputs["attention_mask"] = torch.ones(
            self.config.batch_size,
            self.config.max_seq_length,
            device=device,
            dtype=torch.long,
        )
        
        return inputs
    
    def _simplify_onnx(self, model_path: str):
        """Simplify ONNX model using onnx-simplifier."""
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(model_path)
            model_simp, check = simplify(model)
            
            if check:
                onnx.save(model_simp, model_path)
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification failed, keeping original")
                
        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping simplification")
    
    def _quantize_onnx(self, model_path: str) -> str:
        """Quantize ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = model_path.replace(".onnx", "_quantized.onnx")
            
            quantize_dynamic(
                model_path,
                quantized_path,
                weight_type=QuantType.QInt8,
            )
            
            logger.info(f"Model quantized: {quantized_path}")
            return quantized_path
            
        except ImportError:
            logger.warning("onnxruntime not installed, skipping quantization")
            return model_path
    
    def _save_metadata(self, model_path: str, format: str):
        """Save model metadata alongside exported model."""
        metadata = {
            "format": format,
            "image_size": list(self.config.image_size),
            "max_seq_length": self.config.max_seq_length,
            "opset_version": self.config.opset_version if format == "onnx" else None,
            "quantized": self.config.quantize,
        }
        
        # Add tokenizer vocab size if available
        if self.tokenizer is not None:
            metadata["vocab_size"] = len(self.tokenizer)
        
        # Save metadata
        metadata_path = Path(model_path).with_suffix(".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    tokenizer=None,
    **kwargs,
) -> str:
    """Convenience function for ONNX export."""
    exporter = ModelExporter(model, tokenizer, **kwargs)
    return exporter.export_onnx(output_path)


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    **kwargs,
) -> str:
    """Convenience function for TorchScript export."""
    exporter = ModelExporter(model, **kwargs)
    return exporter.export_torchscript(output_path)
