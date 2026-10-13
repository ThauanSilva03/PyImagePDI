import numpy as np
from core.utils import clip_to_uint8, pad_image

def roberts_cross(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape
    padded = pad_image(img, 1, 1, mode="reflect")
    
    r1 = np.zeros((h, w), dtype=np.float64)
    r2 = np.zeros((h, w), dtype=np.float64)
    
    for i in range(h):
        for j in range(w):
            p_ij = padded[i + 1, j + 1]
            p_next_next = padded[i + 2, j + 2]
            p_next_j = padded[i + 2, j + 1]
            p_i_next = padded[i + 1, j + 2]
            
            r1[i, j] = p_ij - p_next_next
            r2[i, j] = p_i_next - p_next_j
            
    mag = np.sqrt(r1**2 + r2**2)
    return clip_to_uint8(np.abs(r1)), clip_to_uint8(np.abs(r2)), clip_to_uint8(mag)


def sobel_operator(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape
    padded = pad_image(img, 1, 1, mode="reflect")
    
    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)
    
    for i in range(h):
        for j in range(w):
            win = padded[i:i + 3, j:j + 3]
            
            val_gx = (
                -1.0 * win[0, 0] + 1.0 * win[0, 2] +
                -2.0 * win[1, 0] + 2.0 * win[1, 2] +
                -1.0 * win[2, 0] + 1.0 * win[2, 2]
            )
            val_gy = (
                -1.0 * win[0, 0] - 2.0 * win[0, 1] - 1.0 * win[0, 2] +
                 1.0 * win[2, 0] + 2.0 * win[2, 1] + 1.0 * win[2, 2]
            )
            
            gx[i, j] = val_gx
            gy[i, j] = val_gy
            
    mag = np.sqrt(gx**2 + gy**2)
    return clip_to_uint8(np.abs(gx)), clip_to_uint8(np.abs(gy)), clip_to_uint8(mag)


def kirsch_operator(img: np.ndarray) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    h, w = img.shape
    padded = pad_image(img, 1, 1, mode="reflect")
    
    kernels = {
        "N":  np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float64),
        "NO": np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float64),
        "O":  np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float64),
        "SO": np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float64),
        "S":  np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float64),
        "SE": np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float64),
        "E":  np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float64),
        "NE": np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float64),
    }
    
    res_direcionais = {k: np.zeros((h, w), dtype=np.float64) for k in kernels}
    res_final = np.zeros((h, w), dtype=np.float64)
    
    for i in range(h):
        for j in range(w):
            win = padded[i:i + 3, j:j + 3]
            max_val = 0.0
            
            for nome, k_mat in kernels.items():
                soma = 0.0
                for r in range(3):
                    for c in range(3):
                        soma += win[r, c] * k_mat[r, c]
                        
                val_abs = abs(soma)
                res_direcionais[nome][i, j] = val_abs
                if val_abs > max_val:
                    max_val = val_abs
                    
            res_final[i, j] = max_val
            
    lista_direcionais = [(k, clip_to_uint8(v)) for k, v in res_direcionais.items()]
    return lista_direcionais, clip_to_uint8(res_final)


def laplacian_edge(img: np.ndarray, include_diagonals: bool = False) -> np.ndarray:
    h, w = img.shape
    padded = pad_image(img, 1, 1, mode="reflect")
    out = np.zeros((h, w), dtype=np.float64)
    
    if not include_diagonals:
        kernel = np.array([[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]], dtype=np.float64)
    else:
        kernel = np.array([[ 1,  1,  1],
                           [ 1, -8,  1],
                           [ 1,  1,  1]], dtype=np.float64)
                           
    for i in range(h):
        for j in range(w):
            win = padded[i:i + 3, j:j + 3]
            soma = 0.0
            for r in range(3):
                for c in range(3):
                    soma += win[r, c] * kernel[r, c]
            out[i, j] = abs(soma)
            
    return clip_to_uint8(out)