import cv2
import numpy as np
from PIL import Image
import io

def preprocess_signature(image_input, target_height=155, target_width=220):
    """
    Preprocesa una imagen de firma:
    - Convierte a escala de grises
    - Aplica binarización adaptativa
    - Recorta la región de la firma
    - Redimensiona manteniendo proporción
    - Centra la firma en un lienzo blanco de tamaño fijo
    
    Args:
        image_input: Puede ser una ruta de archivo (str) o un stream de archivo
    """
    if isinstance(image_input, str):
        # Es una ruta de archivo
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_input}")
    else:
        # Es un stream de archivo
        try:
            # Leer el stream y convertir a array numpy
            image_input.seek(0)  # Asegurar que estamos al inicio del stream
            pil_image = Image.open(image_input)
            # Convertir a escala de grises si es necesario
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            img = np.array(pil_image)
        except Exception as e:
            raise ValueError(f"Error procesando el stream de imagen: {e}")

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