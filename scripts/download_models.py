#!/usr/bin/env python3
"""Download InsightFace models for offline use"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app import FaceAnalysis


def download_models():
    """Download InsightFace models"""
    print("Downloading InsightFace models...")
    
    models = [
        "buffalo_l",  # Large model (best accuracy)
        "buffalo_m",  # Medium model
        "buffalo_s",  # Small model (fastest)
    ]
    
    for model_name in models:
        print(f"\nDownloading {model_name}...")
        try:
            # Initialize will trigger download
            app = FaceAnalysis(
                name=model_name,
                providers=['CPUExecutionProvider'],
                download=True
            )
            app.prepare(ctx_id=0)
            print(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
    
    # Print model location
    home = Path.home()
    model_dir = home / ".insightface" / "models"
    print(f"\nModels downloaded to: {model_dir}")
    
    if model_dir.exists():
        print("Available models:")
        for model_path in model_dir.iterdir():
            if model_path.is_dir():
                print(f"  - {model_path.name}")


if __name__ == "__main__":
    download_models()