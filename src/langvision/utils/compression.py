
"""
Adaptive Image Compression utilities for simulating network conditions.
"""

from enum import Enum
from typing import Tuple, Optional, Union
from PIL import Image
import io
import logging

class NetworkProfile(str, Enum):
    """Network profiles for adaptive compression."""
    SLOW_2G = "2g"
    FAST_3G = "3g"
    FOUR_G = "4g"
    WIFI = "wifi"

class AdaptiveCompressor:
    """
    Compresses images based on simulated network conditions.
    Used for:
    1. Simulating real-world data degradation during training (robustness).
    2. Optimizing bandwidth if loading from remote URLs.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_settings_for_profile(profile: NetworkProfile) -> dict:
        """Get compression settings for a network profile."""
        if profile == NetworkProfile.SLOW_2G:
            return {"max_size": 224, "quality": 50, "format": "WEBP"}
        elif profile == NetworkProfile.FAST_3G:
            return {"max_size": 480, "quality": 65, "format": "WEBP"}
        elif profile == NetworkProfile.FOUR_G:
            return {"max_size": 1080, "quality": 80, "format": "JPEG"}
        else: # WIFI / HQ
            return {"max_size": None, "quality": 95, "format": "JPEG"}

    def optimize_for_network(self, image: Image.Image, profile: NetworkProfile) -> Image.Image:
        """
        Compress image based on network profile.
        Returns a new PIL Image object (simulating the degraded version).
        """
        settings = self.get_settings_for_profile(profile)
        
        # 1. Resize
        if settings["max_size"]:
            w, h = image.size
            if max(w, h) > settings["max_size"]:
                scale = settings["max_size"] / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 2. Compress (Encode -> Decode to simulate artifacts)
        buffer = io.BytesIO()
        image.save(buffer, format=settings["format"], quality=settings["quality"])
        size_bytes = buffer.tell()
        
        buffer.seek(0)
        degraded_image = Image.open(buffer).convert("RGB")
        
        # self.logger.debug(f"Compressed {profile}: {size_bytes/1024:.1f}KB")
        return degraded_image

    def estimate_compression_ratio(self, original: Image.Image, profile: NetworkProfile) -> float:
        """Estimate compression ratio for a given profile."""
        # Save original to buffer for size reference (assuming JPEG 95)
        orig_buffer = io.BytesIO()
        original.save(orig_buffer, format="JPEG", quality=95)
        orig_size = orig_buffer.tell()
        
        # Compress
        img_deg = self.optimize_for_network(original, profile)
        
        # We need the size of the degraded image stream
        # optimize_for_network returns an Image, we need to save it again with same settings to measure
        settings = self.get_settings_for_profile(profile)
        deg_buffer = io.BytesIO()
        img_deg.save(deg_buffer, format=settings["format"], quality=settings["quality"])
        deg_size = deg_buffer.tell()
        
        return orig_size / deg_size if deg_size > 0 else 1.0
