
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_signature
from utils import euclidean_distance
import sys

# Ruta al modelo entrenado
MODEL_PATH = "modelo_siamesa.h5"

# Rutas a las dos imágenes que quieres comparar
IMG_1_PATH = "../dataset/signatures/full_org/original_25_1.png"
IMG_2_PATH = "../dataset/signatures/full_org/original_1_1.png"

# Cargar el modelo con las funciones personalizadas
model = load_model(MODEL_PATH, custom_objects={"euclidean_distance": euclidean_distance})

# Preprocesar imágenes
img1 = preprocess_signature(IMG_1_PATH)
img2 = preprocess_signature(IMG_2_PATH)

# Expandir a batch size 1
img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)

# Predecir similitud
similarity = model.predict([img1, img2])[0][0]

print(f"🧠 Similitud predicha: {similarity:.4f}")
if similarity > 0.5: # 
    print("✅ Las firmas parecen del MISMO autor.")
else:
    print("❌ Las firmas parecen de AUTORES DISTINTOS.")
