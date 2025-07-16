import cv2, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

print("\n🚀 Ejecutando comparación con el método mejorado...")
print("📖 Leyenda:")
print("   🆕 = Método mejorado (L2 normalización + distancia coseno)")
print("   🔴 = Método anterior (distancia Manhattan)")
print()
# ---------  carga del modelo de embedding (sin Lambda) -------------
embedding_model = tf.keras.models.load_model(
    "signature_embedding_model.keras", compile=False)

THRESHOLD = 0.55
SIZE = 128 

def similarity_score_old(emb1, emb2):
    """Manhattan‑based similarity ∈ [0,1]. 1 = idéntico (MÉTODO ANTERIOR)"""
    return 1 - np.sum(np.abs(emb1 - emb2)) / np.sum(np.abs(emb1) + np.abs(emb2))

def similarity_score(emb1, emb2):
    """
    Distancia Euclidiana normalizada
    Retorna similitud ∈ [0,1] donde 1 = idéntico
    """
    # Calcular distancia euclidiana
    euclidean_dist = np.linalg.norm(emb1 - emb2, axis=1)
    
    # Normalizar a [0,1] - invertir para que 1 = similar
    # Usamos una función exponencial para convertir distancia a similitud
    similarity = np.exp(-euclidean_dist / 10.0)  # Ajustar divisor según sea necesario
    
    return similarity

def preprocess(path, size=SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size)).astype("float32") / 255.0
    return img[..., None]                     # (H,W,1)

def test_signature_pair(ref_path, test_path, threshold=0.5):
    """
    Función simple para probar un par de firmas manualmente
    """
    print(f"  COMPARANDO FIRMAS:")
    print(f"   Original: {ref_path}")
    print(f"   Prueba: {test_path}")
    print("-" * 60)
    
    # Verificar que los archivos existen
    if not os.path.exists(ref_path):
        print(f" No encontrado: {ref_path}")
        return
    if not os.path.exists(test_path):
        print(f" No encontrado: {test_path}")
        return
    
    # Procesar imágenes
    ref_img = preprocess(ref_path)
    test_img = preprocess(test_path)
    
    # Generar embeddings
    ref_emb = embedding_model.predict(ref_img[None, ...], verbose=0)
    test_emb = embedding_model.predict(test_img[None, ...], verbose=0)
    
    # Calcular similitudes con ambos métodos
    score_euclidean = similarity_score(ref_emb, test_emb)[0]
    score_manhattan = similarity_score_old(ref_emb, test_emb)
    
    # Calcular distancia euclidiana cruda
    euclidean_dist_raw = np.linalg.norm(ref_emb - test_emb)
    
    # Resultados
    result_euclidean = "✅ GENUINA" if score_euclidean > threshold else "❌ FALSIFICADA"
    result_manhattan = "✅ GENUINA" if score_manhattan > 0.8 else "❌ FALSIFICADA"
    
    print(f"� RESULTADOS:")
    print(f"   ✅ Distancia Euclidiana:")
    print(f"      - Similitud: {score_euclidean:.6f}")
    print(f"      - Distancia cruda: {euclidean_dist_raw:.6f}")
    print(f"      - Umbral: {threshold}")
    print(f"      - Resultado: {result_euclidean}")
    print()
    print(f"   ❌ Manhattan (anterior):")
    print(f"      - Similitud: {score_manhattan:.6f}")
    print(f"      - Umbral: 0.800")
    print(f"      - Resultado: {result_manhattan}")
    print()
    
    if result_euclidean != result_manhattan:
        print(f"⚠️  Los métodos DIFIEREN!")
    else:
        print(f"✅ Ambos métodos coinciden")
    
    print("=" * 60)

print("SISTEMA DE VERIFICACIÓN CON DISTANCIA EUCLIDIANA")
print("=" * 60)

# CASO 1: Mismo autor (debería ser GENUINA)
print("\n🔬 CASO 1: Firmas del mismo autor")
test_signature_pair(
    "../dataset/signatures/full_org/original_50_1.png",
    "../dataset/signatures/full_org/original_50_2.png"
)

# CASO 2: Original vs Falsificación (debería ser FALSIFICADA)
print("\n🔬 CASO 2: Original vs Falsificación")
test_signature_pair(
    "../dataset/signatures/full_org/original_50_1.png",
    "../dataset/signatures/full_forg/forgeries_50_1.png"
)

print("\n" + "=" * 60)
print("📝 INSTRUCCIONES PARA USO MANUAL:")
print("   Para probar otros pares, llama a:")
print("   test_signature_pair('ruta_firma1.png', 'ruta_firma2.png')")
print("   ")
print("   Puedes ajustar el umbral (por defecto 0.5):")
print("   test_signature_pair('firma1.png', 'firma2.png', threshold=0.7)")
print("=" * 60)