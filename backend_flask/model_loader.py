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


# Variables globales para el modelo
_model = None
_embedding_model = None

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

def get_embedding_model():
    """
    Carga el modelo de embedding para usar método mejorado.
    """
    global _embedding_model
    if _embedding_model is None:
        EMBEDDING_MODEL_PATH = "../modelo_siamesa/signature_embedding_model.keras"
        print(f"🔄 Cargando modelo de embedding desde {EMBEDDING_MODEL_PATH}...")
        try:
            _embedding_model = tf.keras.models.load_model(EMBEDDING_MODEL_PATH, compile=False)
            print("✅ Modelo de embedding cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando modelo de embedding: {e}")
            print("🔄 Usando modelo siamesa tradicional como fallback")
            _embedding_model = None
    return _embedding_model

def similarity_score_improved(emb1, emb2):
    """
    Distancia Euclidiana normalizada (REPLICANDO test_model4.py)
    Retorna similitud ∈ [0,1] donde 1 = idéntico
    """
    # Calcular distancia euclidiana
    euclidean_dist = np.linalg.norm(emb1 - emb2, axis=1)
    
    # Normalizar a [0,1] - invertir para que 1 = similar
    # Usamos una función exponencial para convertir distancia a similitud
    similarity = np.exp(-euclidean_dist / 10.0)  # Ajustar divisor según sea necesario
    
    return similarity

def predict_similarity(img1, img2):
    """
    Compara dos imágenes de firma y retorna la similitud (0 a 1).
    VERSIÓN ACTUALIZADA: Replica la lógica de test_model4.py con distancia euclidiana
    
    Args:
        img1: Array numpy de la primera imagen ya preprocesada
        img2: Array numpy de la segunda imagen ya preprocesada
    """
    # Usar modelo de embedding (como en test_model4.py)
    embedding_model = get_embedding_model()
    
    if embedding_model is not None:
        # MÉTODO EUCLIDIANO: Replicando test_model4.py
        try:
            # Generar embeddings
            if len(img1.shape) == 3:
                img1 = np.expand_dims(img1, axis=0)
            if len(img2.shape) == 3:
                img2 = np.expand_dims(img2, axis=0)
            
            emb1 = embedding_model.predict(img1, verbose=0)
            emb2 = embedding_model.predict(img2, verbose=0)
            
            # Calcular similitud euclidiana (como en test_model4.py)
            similarity = similarity_score_improved(emb1, emb2)[0]
            
            print(f"🆕 Similitud Euclidiana: {similarity:.6f}")
            
            # Retornar similitud euclidiana
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Error con modelo de embedding: {e}")
            print("🔄 Usando modelo siamesa tradicional como fallback")
    
    # FALLBACK: Usar modelo siamesa tradicional
    model = get_model()
    
    # Añadir batch dimension si no la tienen
    if len(img1.shape) == 3:
        img1 = np.expand_dims(img1, axis=0)
    if len(img2.shape) == 3:
        img2 = np.expand_dims(img2, axis=0)

    similarity = model.predict([img1, img2])[0][0]
    print(f"� Similitud tradicional (fallback): {similarity:.6f}")
    return float(similarity)


