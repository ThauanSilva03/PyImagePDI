import numpy as np
from core.utils import clip_to_uint8, pad_image

def adaptive_contrast_enhancement(
    img: np.ndarray, 
    kernel_size: int = 7, 
    c: float = 1.0, 
    border_mode: str = "reflect"
) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("A dimensão da vizinhança deve ser ímpar.")

    h, w = img.shape
    pad = kernel_size // 2
    total_elements = kernel_size * kernel_size

    padded = pad_image(img, pad, pad, mode=border_mode)
    out = np.zeros((h, w), dtype=np.float64)

    global_mean = float(np.mean(img))
    global_std = float(np.std(img))
    eps = 1e-4

    for i in range(h):
        for j in range(w):
            win = padded[i:i + kernel_size, j:j + kernel_size]
            
            local_sum = 0.0
            for r in range(kernel_size):
                for col in range(kernel_size):
                    local_sum += win[r, col]
            local_mean = local_sum / total_elements

            var_sum = 0.0
            for r in range(kernel_size):
                for col in range(kernel_size):
                    diff = win[r, col] - local_mean
                    var_sum += diff * diff
            local_std = np.sqrt(var_sum / total_elements)

            gain = (c * global_std) / (local_std + eps)
            
            f_val = img[i, j]
            val = local_mean + gain * (f_val - local_mean)
            out[i, j] = val

    return clip_to_uint8(out)