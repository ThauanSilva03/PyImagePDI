import numpy as np
from core.utils import clip_to_uint8, pad_image

def convolve2d(img: np.ndarray, kernel: np.ndarray, offset: float = 0.0, border_mode: str = "reflect") -> np.ndarray:
    n, m = kernel.shape
    if n % 2 == 0 or m % 2 == 0:
        raise ValueError("As dimensões da máscara (n, m) devem ser ímpares.")

    h, w = img.shape
    pad_h = n // 2
    pad_w = m // 2

    padded_img = pad_image(img, pad_h, pad_w, mode=border_mode)
    out = np.zeros((h, w), dtype=np.float64)

    flipped_kernel = np.flip(kernel)

    for i in range(h):
        for j in range(w):
            soma = 0.0
            for ki in range(n):
                for kj in range(m):
                    soma += padded_img[i + ki, j + kj] * flipped_kernel[ki, kj]
            
            out[i, j] = soma + offset

    return clip_to_uint8(out)

def get_sharpening_kernel(c: int, d: int) -> np.ndarray:
    return np.array([
        [-c, -c, -c],
        [-c, 8 * c + d, -c],
        [-c, -c, -c]
    ], dtype=np.float64)

def get_emboss_kernels() -> tuple[np.ndarray, np.ndarray]:
    h1 = np.array([
        [0.0, 0.0,  0.0],
        [0.0, 1.0,  0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float64)

    h2 = np.array([
        [ 0.0,  0.0,  2.0],
        [ 0.0, -1.0,  0.0],
        [-1.0,  0.0,  0.0]
    ], dtype=np.float64)

    return h1, h2

def get_mandatory_edge_kernels() -> tuple[np.ndarray, np.ndarray]:
    h3 = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0,  0.0, 1.0],
        [ 0.0,  1.0, 1.0]
    ], dtype=np.float64)

    h4 = np.array([
        [ 0.0, -1.0,  0.0],
        [-1.0, -4.0, -1.0],
        [ 0.0, -1.0,  0.0]
    ], dtype=np.float64)

    return h3, h4