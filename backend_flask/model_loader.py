import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from preprocess import preprocess_signature


def euclidean_distance(vectors):
    """
    Función personalizada usada en el modelo siamesa.
    """
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def l2_normalize_layer(t):
    """
    Función para normalización L2 que se puede usar como custom object.
    """
    return K.l2_normalize(t, axis=1)


# Variable global para el modelo
_model = None

def get_model():
    """
    Carga el modelo de manera lazy (solo cuando se necesite).
    """
    global _model
    if _model is None:
        MODEL_PATH = "modelo_siamesa.h5"
        print(f"🔄 Cargando modelo desde {MODEL_PATH}...")
        custom_objects = {
            "euclidean_distance": euclidean_distance,
            "l2_normalize_layer": l2_normalize_layer
        }
        _model = load_model(MODEL_PATH, custom_objects=custom_objects)
        print("✅ Modelo cargado exitosamente")
    return _model


def predict_similarity(img1, img2):
    """
    Compara dos imágenes de firma y retorna la similitud (0 a 1).
    
    Args:
        img1: Array numpy de la primera imagen ya preprocesada
        img2: Array numpy de la segunda imagen ya preprocesada
    """
    model = get_model()
    
    # Añadir batch dimension si no la tienen
    if len(img1.shape) == 3:
        img1 = np.expand_dims(img1, axis=0)
    if len(img2.shape) == 3:
        img2 = np.expand_dims(img2, axis=0)

    similarity = model.predict([img1, img2])[0][0]
    return float(similarity)


