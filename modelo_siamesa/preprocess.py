import cv2
import numpy as np

def preprocess_signature(image_path, target_height=155, target_width=220):
    """
    Preprocesa una imagen de firma:
    - Convierte a escala de grises
    - Aplica binarización adaptativa
    - Recorta la región de la firma
    - Redimensiona manteniendo proporción
    - Centra la firma en un lienzo blanco de tamaño fijo
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    # Binarización adaptativa para mejorar detección del trazo
    img_bin = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=8
    )

    coords = cv2.findNonZero(img_bin)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_bin[y:y+h, x:x+w]

    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((target_height, target_width), dtype='uint8') * 255
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    canvas = canvas.astype('float32') / 255.0
    canvas = np.expand_dims(canvas, axis=-1)
    return canvas