import numpy as np
from core.utils import clip_to_uint8

def compute_histogram(img: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    h, w = img.shape
    
    for i in range(h):
        for j in range(w):
            val = int(img[i, j])
            hist[val] += 1
            
    return hist

def expand_histogram(img: np.ndarray) -> np.ndarray:
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

def equalize_histogram(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    total_pixels = h * w
    
    hist = compute_histogram(img)
    
    cdf = np.zeros(256, dtype=np.float64)
    acum = 0.0
    for k in range(256):
        acum += hist[k]
        cdf[k] = acum
        
    cdf_min = 0.0
    for k in range(256):
        if cdf[k] > 0:
            cdf_min = cdf[k]
            break
            
    lut = np.zeros(256, dtype=np.uint8)
    denominador = total_pixels - cdf_min
    
    if denominador == 0:
        return img.astype(np.uint8)
        
    for k in range(256):
        if cdf[k] >= cdf_min:
            valor = round(((cdf[k] - cdf_min) / denominador) * 255.0)
            lut[k] = np.clip(valor, 0, 255)
        else:
            lut[k] = 0
            
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = lut[int(img[i, j])]
            
    return out