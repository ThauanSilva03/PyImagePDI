import numpy as np
from core.utils import clip_to_uint8

def negative_transformation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[i, j] = 255 - img[i,j]

    return clip_to_uint8(out)

def threshold_transformation(img: np.ndarray, threshold: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[i,j] = 255.0 if img[i,j] >= threshold else 0.0

    return clip_to_uint8(out)

def contrast_stretching(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    r_min = float(np.min(img))
    r_max = float(np.max(img))

    if r_max == r_min:
        return img.astype(np.uint8)

    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[i, j] = ((img[i, j] - r_min) / (r_max - r_min)) * 255.0

    return clip_to_uint8(out)

def gamma_transformation(img: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            r_norm = img[i, j]/255.0
            out[i,j] = 255.0 * c * (r_norm ** gamma)

    return clip_to_uint8(out)

def log_transformation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    r_max = float(np.max(img))
    if r_max == 0:
        return img.astype(np.uint8)

    c = 255.0 / np.log(1.0 + r_max)
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[i,j] = c * np.log(1.0 + img[i,j])

    return clip_to_uint8(out)