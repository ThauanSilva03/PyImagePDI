# Filtros espaciais de suavização (Média e Mediana)

import numpy as np
from core.utils import clip_to_uint8, pad_image

def mean_filter(img: np.ndarray, kernel_size: int = 3, border_mode: str = "reflect") -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho da vizinhança deve ser um número ímpar.")
        
    h, w = img.shape
    pad = kernel_size // 2
    
    padded_img = pad_image(img, pad, pad, mode=border_mode)
    out = np.zeros((h, w), dtype=np.float64)
    total_elements = kernel_size * kernel_size
    
    for i in range(h):
        for j in range(w):
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            
            soma = 0.0
            for wi in range(kernel_size):
                for wj in range(kernel_size):
                    soma += window[wi, wj]
                    
            out[i, j] = soma / total_elements
            
    return clip_to_uint8(out)

def median_filter(img: np.ndarray, kernel_size: int = 3, border_mode: str = "reflect") -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("O tamanho da vizinhança deve ser um número ímpar.")
        
    h, w = img.shape
    pad = kernel_size // 2
    
    padded_img = pad_image(img, pad, pad, mode=border_mode)
    out = np.zeros((h, w), dtype=np.float64)
    median_idx = (kernel_size * kernel_size) // 2
    
    for i in range(h):
        for j in range(w):
            window = padded_img[i:i + kernel_size, j:j + kernel_size]
            
            elements = []
            for wi in range(kernel_size):
                for wj in range(kernel_size):
                    elements.append(window[wi, wj])
                    
            elements.sort()
            out[i, j] = elements[median_idx]
            
    return clip_to_uint8(out)