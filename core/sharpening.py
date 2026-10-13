# Aguçamento Laplaciano e Filtragem High-Boost

import numpy as np
from core.utils import clip_to_uint8, pad_image

def laplacian_filter_raw(img: np.ndarray, include_diagonals: bool = False) -> np.ndarray:
    h, w = img.shape
    padded = pad_image(img, 1, 1, mode="reflect")
    lap_raw = np.zeros((h, w), dtype=np.float64)

    if not include_diagonals:
        kernel = np.array([
            [ 0.0,  1.0,  0.0],
            [ 1.0, -4.0,  1.0],
            [ 0.0,  1.0,  0.0]
        ], dtype=np.float64)
    else:
        kernel = np.array([
            [ 1.0,  1.0,  1.0],
            [ 1.0, -8.0,  1.0],
            [ 1.0,  1.0,  1.0]
        ], dtype=np.float64)

    for i in range(h):
        for j in range(w):
            win = padded[i:i + 3, j:j + 3]
            soma = 0.0
            for r in range(3):
                for c in range(3):
                    soma += win[r, c] * kernel[r, c]
            lap_raw[i, j] = soma

    return lap_raw

def laplacian_sharpening(img: np.ndarray, include_diagonals: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lap_raw = laplacian_filter_raw(img, include_diagonals)
    
    sharpened_float = img - lap_raw
    sharpened = clip_to_uint8(sharpened_float)
    
    lap_visual = clip_to_uint8(lap_raw + 128.0)
    
    return lap_visual, sharpened

def high_boost_filtering(img: np.ndarray, A: float = 1.2, include_diagonals: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lap_raw = laplacian_filter_raw(img, include_diagonals)
    
    high_boost_float = (A * img) - lap_raw
    high_boost = clip_to_uint8(high_boost_float)
    
    lap_visual = clip_to_uint8(lap_raw + 128.0)
    
    return lap_visual, high_boost