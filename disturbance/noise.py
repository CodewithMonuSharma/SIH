import cv2
import numpy as np
from typing import Optional

class ImageNoiseEngine:
    """
    FR-10 Noise Injection Subsystem.
    Applies Gaussian, Salt & Pepper (~10%), and/or Poisson noise to image frames.
    Allows toggling individual noise types and setting intensity levels.
    """
    def __init__(self, random_seed: Optional[int] = 42):
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def apply_gaussian_noise(self, image: np.ndarray, std_dev: float = 15.0) -> np.ndarray:
        """Applies Gaussian distributed per-pixel intensity noise (std dev <= 20 px)."""
        if std_dev <= 0:
            return image
        noise = self.rng.normal(0, std_dev, image.shape).astype(np.float32)
        noisy_img = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_img

    def apply_salt_and_pepper_noise(self, image: np.ndarray, density: float = 0.05) -> np.ndarray:
        """Applies Salt & Pepper impulse noise (density ~0.05 - 0.10)."""
        if density <= 0:
            return image
        noisy_img = image.copy()
        num_pixels = int(density * image.size)

        # Salt (white pixels)
        coords_salt = [self.rng.integers(0, i, num_pixels) for i in image.shape[:2]]
        noisy_img[tuple(coords_salt)] = 255

        # Pepper (black pixels)
        coords_pepper = [self.rng.integers(0, i, num_pixels) for i in image.shape[:2]]
        noisy_img[tuple(coords_pepper)] = 0

        return noisy_img

    def apply_poisson_noise(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Applies Poisson (shot) signal-dependent noise."""
        if scale <= 0:
            return image
        # Scale values to represent photon counts
        scaled = np.clip(image.astype(np.float32) / 255.0 * 100.0 * scale, 0, None)
        noisy_scaled = self.rng.poisson(scaled).astype(np.float32)
        noisy_img = np.clip(noisy_scaled / (100.0 * scale) * 255.0, 0, 255).astype(np.uint8)
        return noisy_img
