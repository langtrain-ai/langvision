"""
Langvision API Client

Lightweight client for communicating with Langvision server.
All heavy operations (training, inference, evaluation) run on the server.

The client handles:
- Authentication with API keys
- Job submission and monitoring
- Result retrieval and streaming
- Local preprocessing before upload
"""

import os
import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Union, Iterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

# HTTP client - use requests if available, fallback to urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ServerConfig:
    """Configuration for server connection."""
    base_url: str = "https://api.langtrain.xyz"
    api_version: str = "v1"
    timeout: int = 30
    max_retries: int = 3
    
    # Streaming
    stream_timeout: int = 300
    
    # Uploads
    chunk_size: int = 8 * 1024 * 1024  # 8MB chunks
    max_file_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    
    @property
    def api_url(self) -> str:
        return f"{self.base_url}/{self.api_version}"


@dataclass
class JobResult:
    """Result from a server job."""
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
    
    @property
    def is_success(self) -> bool:
        return self.status == JobStatus.COMPLETED


class LangvisionAPIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: int = None, response: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(LangvisionAPIError):
    """Authentication failed."""
    pass


class RateLimitError(LangvisionAPIError):
    """Rate limit exceeded."""
    pass


class ServerError(LangvisionAPIError):
    """Server-side error."""
    pass


class LangvisionClient:
    """
    Client for Langvision server API.
    
    All heavy operations (training, inference, evaluation) are executed
    on the server. This client handles:
    - Job submission
    - Progress monitoring
    - Result streaming
    - File uploads/downloads
    
    Usage:
        client = LangvisionClient(api_key="lv-...")
        
        # Submit training job
        job = client.submit_training(
            model="llava-v1.6-7b",
            dataset_id="my-dataset",
            config={...}
        )
        
        # Wait for completion
        result = client.wait_for_job(job.job_id)
        
        # Or monitor progress
        for update in client.stream_job_progress(job.job_id):
            print(f"Progress: {update.progress}%")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ServerConfig] = None,
    ):
        self.config = config or ServerConfig()
        
        # Get API key from param, env, or config file
        self.api_key = api_key or self._get_api_key()
        
        if not self.api_key:
            logger.warning(
                "No API key provided. Set LANGVISION_API_KEY environment variable "
                "or run 'langvision auth login'"
            )
        
        self._session = None
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or config file."""
        # Check environment
        api_key = os.environ.get("LANGVISION_API_KEY")
        if api_key:
            return api_key
        
        # Check config file
        config_path = Path.home() / ".langvision" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    return config.get("api_key")
            except:
                pass
        
        return None
    
    def _get_session(self):
        """Get or create HTTP session."""
        if HAS_REQUESTS:
            if self._session is None:
                self._session = requests.Session()
                self._session.headers.update({
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "langvision-python/0.1.0",
                })
            return self._session
        return None
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Make HTTP request to server."""
        url = f"{self.config.api_url}/{endpoint}"
        
        if HAS_REQUESTS:
            session = self._get_session()
            
            try:
                if files:
                    # Multipart upload
                    response = session.request(
                        method,
                        url,
                        data=data,
                        files=files,
                        params=params,
                        timeout=self.config.timeout,
                        stream=stream,
                    )
                else:
                    response = session.request(
                        method,
                        url,
                        json=data,
                        params=params,
                        timeout=self.config.timeout,
                        stream=stream,
                    )
                
                if stream:
                    return response
                
                return self._handle_response(response)
                
            except requests.exceptions.Timeout:
                raise LangvisionAPIError("Request timed out")
            except requests.exceptions.ConnectionError:
                raise LangvisionAPIError("Failed to connect to server")
        else:
            # Fallback to urllib
            return self._urllib_request(method, url, data, params)
    
    def _urllib_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict],
        params: Optional[Dict],
    ) -> Dict[str, Any]:
        """Fallback HTTP request using urllib."""
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(params)}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "langvision-python/0.1.0",
        }
        
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            try:
                error_data = json.loads(error_body)
            except:
                error_data = {"message": error_body}
            
            self._handle_error(e.code, error_data)
    
    def _handle_response(self, response) -> Dict[str, Any]:
        """Handle HTTP response."""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 201:
            return response.json()
        elif response.status_code == 204:
            return {}
        else:
            try:
                error_data = response.json()
            except:
                error_data = {"message": response.text}
            
            self._handle_error(response.status_code, error_data)
    
    def _handle_error(self, status_code: int, error_data: Dict):
        """Handle error response."""
        message = error_data.get("message", "Unknown error")
        
        if status_code == 401:
            raise AuthenticationError(message, status_code, error_data)
        elif status_code == 429:
            raise RateLimitError(message, status_code, error_data)
        elif status_code >= 500:
            raise ServerError(message, status_code, error_data)
        else:
            raise LangvisionAPIError(message, status_code, error_data)
    
    # ==================== Authentication & Validation ====================
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the API key and return plan/feature info.
        
        Returns:
            dict with:
                - valid: bool
                - plan: str (free, pro, enterprise)
                - features: list of feature names
                - limits: dict of limits
                - workspace_id: str
        """
        if not self.api_key:
            return {"valid": False, "error": "No API key configured"}
        
        try:
            response = self._request("POST", "auth/api-keys/validate", data={"api_key": self.api_key})
            return response
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def validate_api_key(self) -> bool:
        """Validate the current API key (simple check)."""
        result = self.validate()
        return result.get("valid", False)
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        return self.validate_api_key()
    
    def get_features(self) -> List[str]:
        """Get list of available features for current plan."""
        result = self.validate()
        return result.get("features", [])
    
    def has_feature(self, feature: str) -> bool:
        """Check if a specific feature is available."""
        return feature in self.get_features()
    
    def get_limits(self) -> Dict[str, int]:
        """Get current plan limits."""
        result = self.validate()
        return result.get("limits", {})
    
    def get_plan(self) -> str:
        """Get current subscription plan name."""
        result = self.validate()
        return result.get("plan", "free")
    
    def get_usage(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self._request("GET", "auth/usage")
    
    # ==================== Training Jobs ====================
    
    def submit_training(
        self,
        model: str,
        dataset_id: str,
        training_method: str = "qlora",
        config: Optional[Dict[str, Any]] = None,
        sft_config: Optional[Dict[str, Any]] = None,
        dpo_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> JobResult:
        """
        Submit a training job to the server.
        
        Args:
            model: Model name (e.g., "llava-v1.6-7b")
            dataset_id: ID of uploaded dataset
            training_method: Training method - one of:
                - "sft" (Supervised Fine-Tuning)
                - "dpo" (Direct Preference Optimization)
                - "rlhf" (Reinforcement Learning from Human Feedback)
                - "lora" (LoRA adapters)
                - "qlora" (Quantized LoRA, default)
            config: Training configuration (epochs, lr, batch_size, etc.)
            sft_config: SFT-specific config
            dpo_config: DPO-specific config
            **kwargs: Additional training options
        
        Returns:
            JobResult with job_id for tracking
        """
        training_config = config or {}
        training_config.update(kwargs)
        
        payload = {
            "model": model,
            "dataset_id": dataset_id,
            "training_method": training_method,
            "config": training_config,
        }
        
        if sft_config and training_method == "sft":
            payload["sft_config"] = sft_config
        if dpo_config and training_method == "dpo":
            payload["dpo_config"] = dpo_config
        
        response = self._request("POST", "training/jobs", data=payload)
        
        return JobResult(
            job_id=response["job_id"],
            status=JobStatus(response["status"]),
            created_at=response.get("created_at"),
        )
    
    def get_job_status(self, job_id: str) -> JobResult:
        """Get current status of a job."""
        response = self._request("GET", f"jobs/{job_id}")
        
        return JobResult(
            job_id=response["job_id"],
            status=JobStatus(response["status"]),
            result=response.get("result"),
            error=response.get("error"),
            progress=response.get("progress", 0),
            logs=response.get("logs", []),
            created_at=response.get("created_at"),
            completed_at=response.get("completed_at"),
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        try:
            self._request("POST", f"jobs/{job_id}/cancel")
            return True
        except LangvisionAPIError:
            return False
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[JobResult], None]] = None,
    ) -> JobResult:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
            callback: Optional callback for progress updates
        
        Returns:
            Final JobResult
        """
        start_time = time.time()
        
        while True:
            result = self.get_job_status(job_id)
            
            if callback:
                callback(result)
            
            if result.is_complete:
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def stream_job_progress(
        self,
        job_id: str,
        poll_interval: float = 2.0,
    ) -> Iterator[JobResult]:
        """
        Stream job progress updates.
        
        Yields:
            JobResult updates until job completes
        """
        while True:
            result = self.get_job_status(job_id)
            yield result
            
            if result.is_complete:
                break
            
            time.sleep(poll_interval)
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[JobResult]:
        """List user's jobs."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        response = self._request("GET", "jobs", params=params)
        
        return [
            JobResult(
                job_id=job["job_id"],
                status=JobStatus(job["status"]),
                progress=job.get("progress", 0),
                created_at=job.get("created_at"),
            )
            for job in response.get("jobs", [])
        ]
    
    # ==================== Inference ====================
    
    def generate(
        self,
        model: str,
        image_url: str,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Generate response from Vision LLM.
        
        Args:
            model: Model name or fine-tuned model ID
            image_url: URL of image to process
            prompt: Text prompt
            **kwargs: Generation options (temperature, max_tokens, etc.)
        
        Returns:
            Generated text response
        """
        payload = {
            "model": model,
            "image_url": image_url,
            "prompt": prompt,
            **kwargs,
        }
        
        response = self._request("POST", "inference/generate", data=payload)
        return response["text"]
    
    def stream_generate(
        self,
        model: str,
        image_url: str,
        prompt: str,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream generation token by token.
        
        Yields:
            Generated tokens
        """
        if not HAS_REQUESTS:
            # Fallback to non-streaming
            yield self.generate(model, image_url, prompt, **kwargs)
            return
        
        payload = {
            "model": model,
            "image_url": image_url,
            "prompt": prompt,
            "stream": True,
            **kwargs,
        }
        
        response = self._request(
            "POST", "inference/generate",
            data=payload,
            stream=True,
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                if "text" in data:
                    yield data["text"]
                if data.get("done"):
                    break
    
    def batch_generate(
        self,
        model: str,
        inputs: List[Dict[str, str]],
        **kwargs,
    ) -> List[str]:
        """
        Batch generation for multiple inputs.
        
        Args:
            model: Model name
            inputs: List of {"image_url": ..., "prompt": ...}
            **kwargs: Generation options
        
        Returns:
            List of generated responses
        """
        payload = {
            "model": model,
            "inputs": inputs,
            **kwargs,
        }
        
        response = self._request("POST", "inference/batch", data=payload)
        return response["results"]
    
    # ==================== Datasets ====================
    
    def upload_dataset(
        self,
        file_path: str,
        name: str,
        description: str = "",
        task: str = "vqa",
    ) -> str:
        """
        Upload a dataset to the server.
        
        Args:
            file_path: Path to dataset file (JSON/JSONL)
            name: Dataset name
            description: Dataset description
            task: Task type (vqa, captioning, preference)
        
        Returns:
            Dataset ID
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        file_size = path.stat().st_size
        if file_size > self.config.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.config.max_file_size})")
        
        # For large files, use chunked upload
        if file_size > self.config.chunk_size:
            return self._chunked_upload(path, name, description, task)
        
        # Small files - single upload
        with open(path, 'rb') as f:
            files = {"file": (path.name, f)}
            data = {"name": name, "description": description, "task": task}
            
            response = self._request("POST", "datasets/upload", data=data, files=files)
        
        return response["dataset_id"]
    
    def _chunked_upload(
        self,
        path: Path,
        name: str,
        description: str,
        task: str,
    ) -> str:
        """Upload large file in chunks."""
        # Initialize upload
        file_size = path.stat().st_size
        file_hash = self._compute_file_hash(path)
        
        init_response = self._request("POST", "datasets/upload/init", data={
            "name": name,
            "description": description,
            "task": task,
            "file_size": file_size,
            "file_hash": file_hash,
            "filename": path.name,
        })
        
        upload_id = init_response["upload_id"]
        
        # Upload chunks
        with open(path, 'rb') as f:
            chunk_num = 0
            while True:
                chunk = f.read(self.config.chunk_size)
                if not chunk:
                    break
                
                files = {"chunk": (f"chunk_{chunk_num}", chunk)}
                self._request(
                    "POST",
                    f"datasets/upload/{upload_id}/chunk",
                    data={"chunk_number": chunk_num},
                    files=files,
                )
                chunk_num += 1
        
        # Complete upload
        complete_response = self._request(
            "POST",
            f"datasets/upload/{upload_id}/complete",
        )
        
        return complete_response["dataset_id"]
    
    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List user's datasets."""
        response = self._request("GET", "datasets")
        return response.get("datasets", [])
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        try:
            self._request("DELETE", f"datasets/{dataset_id}")
            return True
        except LangvisionAPIError:
            return False
    
    # ==================== Models ====================
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available base models."""
        response = self._request("GET", "models")
        return response.get("models", [])
    
    def list_fine_tuned_models(self) -> List[Dict[str, Any]]:
        """List user's fine-tuned models."""
        response = self._request("GET", "models/fine-tuned")
        return response.get("models", [])
    
    def download_model(
        self,
        model_id: str,
        output_dir: str,
        format: str = "safetensors",
    ) -> str:
        """
        Download a fine-tuned model.
        
        Args:
            model_id: Model ID
            output_dir: Directory to save model
            format: Download format (safetensors, pytorch, gguf)
        
        Returns:
            Path to downloaded model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        response = self._request("GET", f"models/{model_id}/download", params={"format": format})
        
        download_url = response["download_url"]
        
        if HAS_REQUESTS:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                file_path = output_path / f"{model_id}.{format}"
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            with urllib.request.urlopen(download_url) as r:
                file_path = output_path / f"{model_id}.{format}"
                with open(file_path, 'wb') as f:
                    while True:
                        chunk = r.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
        
        return str(file_path)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a fine-tuned model."""
        try:
            self._request("DELETE", f"models/{model_id}")
            return True
        except LangvisionAPIError:
            return False
    
    # ==================== Evaluation ====================
    
    def submit_evaluation(
        self,
        model_id: str,
        dataset_id: str,
        task: str = "vqa",
        metrics: Optional[List[str]] = None,
    ) -> JobResult:
        """
        Submit an evaluation job.
        
        Args:
            model_id: Model to evaluate
            dataset_id: Evaluation dataset
            task: Task type
            metrics: Metrics to compute
        
        Returns:
            JobResult for tracking
        """
        payload = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "task": task,
            "metrics": metrics or ["accuracy"],
        }
        
        response = self._request("POST", "evaluation/jobs", data=payload)
        
        return JobResult(
            job_id=response["job_id"],
            status=JobStatus(response["status"]),
        )


# Convenience functions

def get_client(api_key: Optional[str] = None) -> LangvisionClient:
    """Get a configured Langvision client."""
    return LangvisionClient(api_key=api_key)


def train(
    model: str,
    dataset_id: str,
    wait: bool = True,
    **config,
) -> Union[JobResult, str]:
    """
    Submit training job (convenience function).
    
    Args:
        model: Model name
        dataset_id: Dataset ID
        wait: Wait for completion
        **config: Training configuration
    
    Returns:
        JobResult if wait=True, else job_id
    """
    client = get_client()
    job = client.submit_training(model, dataset_id, config)
    
    if wait:
        return client.wait_for_job(
            job.job_id,
            callback=lambda r: print(f"Progress: {r.progress:.1f}%")
        )
    
    return job.job_id


def generate(
    model: str,
    image_url: str,
    prompt: str,
    **kwargs,
) -> str:
    """Generate response (convenience function)."""
    client = get_client()
    return client.generate(model, image_url, prompt, **kwargs)
