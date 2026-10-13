# Dissolve cruzado uniforme e não uniforme (degradê)

import numpy as np
from core.utils import clip_to_uint8

def cross_dissolve_uniform(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError("As imagens devem possuir as mesmas dimensões.")

    h, w = img1.shape
    out = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            out[i, j] = alpha * img1[i, j] + (1.0 - alpha) * img2[i, j]

    return clip_to_uint8(out)

def cross_dissolve_non_uniform(img1: np.ndarray, img2: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError("As imagens devem possuir as mesmas dimensões.")

    h, w = img1.shape
    out = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            if direction == "horizontal":
                alpha = j / (w - 1) if w > 1 else 1.0
            else:
                alpha = i / (h - 1) if h > 1 else 1.0

            out[i, j] = alpha * img1[i, j] + (1.0 - alpha) * img2[i, j]

    return clip_to_uint8(out)