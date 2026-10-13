import numpy as np

def clip_to_uint8(image_float: np.ndarray) -> np.ndarray:
    clipped = np.clip(image_float, 0, 255)
    return clipped.astype(np.uint8)

def pad_image(img: np.ndarray, pad_h: int, pad_w: int, mode: str = "reflect") -> np.ndarray:
    h, w = img.shape
    new_h = h + 2 * pad_h
    new_w = w + 2 * pad_w
    padded = np.zeros((new_h, new_w), dtype=np.float64)

    padded[pad_h:pad_h + h, pad_w:pad_w + w] = img

    mode_norm = str(mode).lower()
    if mode_norm in ("constant", "constants"):
        return padded
    elif mode_norm == "reflect":
        # Bordas superior e inferior
        for i in range(pad_h):
            padded[pad_h - 1 - i, pad_w:pad_w + w] = img[i, :]
            padded[pad_h + h + i, pad_w:pad_w + w] = img[h - 1 - i, :]
        # Bordas laterais e cantos
        for j in range(pad_w):
            padded[:, pad_w - 1 - j] = padded[:, pad_w + j]
            padded[:, pad_w + w + j] = padded[:, pad_w + w - 1 - j]

        return padded
    else:
        raise ValueError("Modo de borda não suportado.")