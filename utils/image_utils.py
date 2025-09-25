import os
import uuid
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


def validate_image(pil_image: Image.Image, min_size: int = 50) -> bool:
    """
    Validate if the image is usable for face recognition.
    - Must be larger than `min_size` in both width and height.
    - Must have 3 channels (RGB).
    """
    if pil_image is None:
        return False

    width, height = pil_image.size
    if width < min_size or height < min_size:
        return False

    cv_img = pil_to_cv2(pil_image)
    if len(cv_img.shape) != 3 or cv_img.shape[2] != 3:
        return False

    return True
