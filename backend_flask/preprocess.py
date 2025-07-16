import cv2
import numpy as np
from PIL import Image
import io

def _is_clean_scan(gray):
    """Heurística rápida: ¿el fondo es casi blanco y homogéneo?"""
    h, w = gray.shape
    # Proporción de píxeles > 240 (muy claros)
    white_ratio = np.sum(gray > 240) / (h * w)
    return white_ratio > 0.70   # 70 % del fondo muy claro ⇒ escaneada

def _deskew_and_crop(gray):
    """Detecta contorno dominante (hoja/firma) y hace warp + crop"""
    # 1. Suavizado y Canny
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # 2. Buscar contornos grandes
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray      # fallback
    # contorno con área máxima
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cropped = gray[y:y+h, x:x+w]
    return cropped

def preprocess_signature_simple(image_input, size=128):
    """
    Pre‑procesamiento robusto:
    - Detecta tipo (scan vs foto)
    - Aplica pipeline extra sólo para fotos
    - Devuelve tensor (H, W, 1) float32 in [0,1]
    
    Args:
        image_input: Puede ser una ruta de archivo (str) o un stream de archivo
        size: Tamaño objetivo (por defecto 128, como en test_model4.py)
    """
    # --- 1. leer imagen en escala de grises ----------------------------
    if isinstance(image_input, str):
        gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"No se pudo cargar {image_input}")
    else:  # stream
        image_input.seek(0)
        pil = Image.open(image_input).convert('L')
        gray = np.array(pil)

    # --- 2. decidir si es foto o scan ----------------------------------
    if _is_clean_scan(gray):
        processed = gray  # Escaneada -> nada extra
    else:
        # ---- pipeline extra para foto ----
        # a) Reducción opcional
        if max(gray.shape) > 1024:
            scale = 1024 / max(gray.shape)
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        # b) Deskew + crop
        gray = _deskew_and_crop(gray)
        # c) Denoise
        gray = cv2.medianBlur(gray, 3)
        # d) CLAHE (mejora contraste)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # e) Binarizar opcional (adaptativo)
        gray = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 17, 15)
        processed = gray

    # --- 3. resize + normalizar ----------------------------------------
    processed = cv2.resize(processed, (size, size), interpolation=cv2.INTER_AREA)
    img = processed.astype('float32') / 255.0
    return img[..., None]          # shape (size,size,1)

def preprocess_signature(image_input, size=128):
    """
    Función de compatibilidad que usa el mismo preprocesamiento robusto.
    Mantiene el nombre original para backward compatibility.
    
    Args:
        image_input: Puede ser una ruta de archivo (str) o un stream de archivo
        size: Tamaño objetivo (por defecto 128)
    """
    return preprocess_signature_simple(image_input, size)