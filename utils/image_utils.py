import os
import uuid
import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to OpenCV (NumPy array in BGR format).
    """
    cv2_image = np.array(pil_image)
    if cv2_image.ndim == 2:  # grayscale
        return cv2_image
    return cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)


def save_image_to_disk(
    pil_image: Image.Image,
    output_dir: str,
    filename: Optional[str] = None,
    ext: str = "jpg",
) -> str:
    """
    Save a PIL Image to disk and return the file path.
    - Creates the output directory if it doesn't exist.
    - Generates a random filename if not provided.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = str(uuid.uuid4())

    file_path = os.path.join(output_dir, f"{filename}.{ext}")
    pil_image.save(file_path)
    return file_path


def validate_image(image_bytes: bytes, min_size: int = 50) -> Image.Image:
    """
    Validate and load an image from raw bytes.
    - Returns a PIL Image if valid, raises ValueError if not.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image data: cannot open as image")

    width, height = pil_image.size
    if width < min_size or height < min_size:
        raise ValueError(f"Image too small: {width}x{height}, min={min_size}")

    cv_img = pil_to_cv2(pil_image)
    if len(cv_img.shape) != 3 or cv_img.shape[2] != 3:
        raise ValueError("Invalid image: must have 3 channels (RGB)")

    return pil_image
