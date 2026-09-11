import cv2
import numpy as np

class AtmosphericConditionEngine:
    """
    FR-11 Atmospheric Condition Simulation Subsystem.
    Simulates Clear, Haze, Fog, Rain, and Low Light as image-level contrast,
    brightness, blur, and streak transforms per PRD Section 9 specifications.
    """
    @staticmethod
    def apply_condition(image: np.ndarray, condition: str = "Clear", severity: float = 0.5) -> np.ndarray:
        condition = condition.strip().capitalize()
        if condition == "Clear":
            return image

        img_float = image.astype(np.float32)

        if condition == "Haze":
            # Contrast reduction + atmospheric veil
            contrast_factor = max(0.3, 1.0 - 0.4 * severity)
            veil_level = 60.0 * severity
            hazy = img_float * contrast_factor + veil_level
            return np.clip(hazy, 0, 255).astype(np.uint8)

        elif condition == "Fog":
            # Stronger contrast reduction + atmospheric blur
            contrast_factor = max(0.15, 1.0 - 0.7 * severity)
            foggy = img_float * contrast_factor + (100.0 * severity)
            blurred = cv2.GaussianBlur(np.clip(foggy, 0, 255).astype(np.uint8), (5, 5), 1.5 * severity)
            return blurred

        elif condition == "Rain":
            # Mild contrast reduction + rain streak overlay
            rainy = (img_float * 0.85).astype(np.uint8)
            h, w = rainy.shape[:2]
            # Draw synthetic slanted rain streaks
            np.random.seed(42)
            num_streaks = int(100 * severity)
            for _ in range(num_streaks):
                rx = np.random.randint(0, w)
                ry = np.random.randint(0, h)
                cv2.line(rainy, (rx, ry), (rx + 3, ry + 12), (180, 180, 180), 1)
            return rainy

        elif condition in ("Lowlight", "Low_light", "Low light", "Night"):
            # Brightness reduction & gamma dimming for NIR night view
            dimmed = img_float * max(0.15, 1.0 - 0.6 * severity)
            return np.clip(dimmed, 0, 255).astype(np.uint8)

        return image
