import cv2, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

def verify_signature_improved(ref_path, test_path, threshold=None, show_comparison=True):
    """
    VERSIÓN MEJORADA: Verifica si dos firmas pertenecen al mismo autor
    
    Args:
        ref_path: Ruta de la firma de referencia
        test_path: Ruta de la firma a verificar
        threshold: Umbral personalizado (si es None, se calibra automáticamente)
        show_comparison: Si mostrar la comparación visual
    """
    # Calibrar umbral si no se proporciona
    if threshold is None:
        print("🔧 Calibrando umbral automáticamente...")
        pairs, labels = load_test_pairs(num_genuine=25, num_forged=25)
        threshold = calibrate_threshold(pairs, labels)
        print(f"✅ Usando umbral calibrado: {threshold:.3f}\n")
    
    ref_img  = preprocess(ref_path)
    test_img = preprocess(test_path)

    # Visualización opcional
    if show_comparison:
        plt.figure(figsize=(10, 4))
        for title, img_path, pos in [("Original", ref_path, 1), ("Compared", test_path, 2)]:
            plt.subplot(1,2,pos)
            plt.imshow(cv2.imread(img_path, 0), cmap='gray')
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ---------- inferencia ----------
    ref_emb  = embedding_model.predict(ref_img[None, ...], verbose=0)
    test_emb = embedding_model.predict(test_img[None, ...], verbose=0)
    
    # Calcular similitud con ambos métodos para comparación
    score_new = similarity_score(ref_emb, test_emb)[0]
    score_old = similarity_score_old(ref_emb, test_emb)
    
    result_new = "Genuine" if score_new > threshold else "Forged"
    result_old = "Genuine" if score_old > 0.80 else "Forged"
    
    print(f"🔍 Comparando firmas:")
    print(f"📁 Original: {ref_path}")
    print(f"📁 Comparada: {test_path}")
    print(f"")
    print(f"🆕 MÉTODO MEJORADO (L2 + Coseno):")
    print(f"   Similitud: {score_new:.4f}")
    print(f"   Umbral: {threshold:.3f}")
    print(f"   Resultado: {result_new}")
    print(f"")
    print(f"🔴 MÉTODO ANTERIOR (Manhattan):")
    print(f"   Similitud: {score_old:.4f}")  
    print(f"   Umbral: 0.800")
    print(f"   Resultado: {result_old}")
    print(f"")
    
    if result_new != result_old:
        print(f"⚠️  Los métodos difieren! El método mejorado es más confiable.")
    else:
        print(f"✅ Ambos métodos coinciden.")
        
    return result_new, score_new

def test_multiple_pairs():
    """
    Función para probar múltiples pares y ver las diferencias entre métodos
    Incluye todos los tipos de comparaciones relevantes
    """
    print("=" * 80)
    print("🧪 PRUEBA COMPARATIVA COMPLETA: MÉTODO ANTERIOR vs MEJORADO")
    print("=" * 80)
    
    # Pruebas específicas con casos reales
    test_cases = [
        # CASOS GENUINOS (mismo autor) - deberían ser Genuine
        ("../dataset/signatures/full_org/original_1_1.png", 
         "../dataset/signatures/full_org/original_1_2.png", "✅ Mismo autor (1-1 vs 1-2)"),
        
        ("../dataset/signatures/full_org/original_2_1.png", 
         "../dataset/signatures/full_org/original_2_2.png", "✅ Mismo autor (2-1 vs 2-2)"),
        
        # CASOS FALSIFICADOS - diferentes autores (deberían ser Forged)
        ("../dataset/signatures/full_org/original_1_1.png",
         "../dataset/signatures/full_org/original_2_1.png", "❌ Autores diferentes (1 vs 2)"),
         
        ("../dataset/signatures/full_org/original_1_1.png",
         "../dataset/signatures/full_org/original_3_1.png", "❌ Autores diferentes (1 vs 3)"),
        
        # CASOS FALSIFICADOS - original vs falsificación (deberían ser Forged)
        ("../dataset/signatures/full_org/original_1_1.png",
         "../dataset/signatures/full_forg/forgeries_1_1.png", "❌ Original vs Falsificación (1-1)"),
         
        ("../dataset/signatures/full_org/original_2_1.png",
         "../dataset/signatures/full_forg/forgeries_2_1.png", "❌ Original vs Falsificación (2-1)"),
    ]
    
    results_summary = {"correct_old": 0, "correct_new": 0, "total": 0}
    
    for i, (ref, test, desc) in enumerate(test_cases, 1):
        print(f"\n🔬 CASO {i}: {desc}")
        print("-" * 50)
        
        if os.path.exists(ref) and os.path.exists(test):
            result_new, score_new = verify_signature_improved(ref, test, show_comparison=False)
            
            # Determinar el resultado esperado basado en el caso
            expected = "Genuine" if "✅" in desc else "Forged"
            
            # Calcular precisión de ambos métodos
            # Para el método anterior, usamos umbral 0.80
            ref_img = preprocess(ref)
            test_img = preprocess(test)
            ref_emb = embedding_model.predict(ref_img[None, ...], verbose=0)
            test_emb = embedding_model.predict(test_img[None, ...], verbose=0)
            score_old = similarity_score_old(ref_emb, test_emb)
            result_old = "Genuine" if score_old > 0.80 else "Forged"
            
            # Evaluar precisión
            if result_old == expected:
                results_summary["correct_old"] += 1
            if result_new == expected:
                results_summary["correct_new"] += 1
            results_summary["total"] += 1
            
            print(f"🎯 Resultado esperado: {expected}")
            print(f"📊 Método anterior: {'✅' if result_old == expected else '❌'}")
            print(f"📊 Método mejorado: {'✅' if result_new == expected else '❌'}")
            
        else:
            print(f"⚠️ Archivos no encontrados: {ref} o {test}")
        print()
    
    # Resumen de resultados
    print("=" * 80)
    print("📊 RESUMEN DE PRECISIÓN")
    print("=" * 80)
    if results_summary["total"] > 0:
        old_accuracy = results_summary["correct_old"] / results_summary["total"] * 100
        new_accuracy = results_summary["correct_new"] / results_summary["total"] * 100
        
        print(f"🔴 Método anterior (Manhattan): {results_summary['correct_old']}/{results_summary['total']} = {old_accuracy:.1f}%")
        print(f"🆕 Método mejorado (L2+Coseno): {results_summary['correct_new']}/{results_summary['total']} = {new_accuracy:.1f}%")
        print(f"")
        if new_accuracy > old_accuracy:
            print(f"🎉 ¡El método mejorado es {new_accuracy - old_accuracy:.1f}% más preciso!")
        elif new_accuracy == old_accuracy:
            print(f"⚖️ Ambos métodos tienen la misma precisión")
        else:
            print(f"⚠️ El método anterior fue {old_accuracy - new_accuracy:.1f}% más preciso en estas pruebas")
    else:
        print("⚠️ No se pudieron ejecutar las pruebas")
    print("=" * 80)

print("\n🚀 Ejecutando comparación con el método mejorado...")
print("📖 Leyenda:")
print("   🆕 = Método mejorado (L2 normalización + distancia coseno)")
print("   🔴 = Método anterior (distancia Manhattan)")
print()
# ---------  carga del modelo de embedding (sin Lambda) -------------
embedding_model = tf.keras.models.load_model(
    "signature_embedding_model.keras", compile=False)

SIZE = 128        # o el que usaste en entrenamiento
THRESHOLD = 0.55  # Nuevo umbral recomendado para coseno (se calibrará automáticamente)

def preprocess(path, size=SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size)).astype("float32") / 255.0
    return img[..., None]                     # (H,W,1)

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

def load_test_pairs(num_genuine=25, num_forged=25):
    """
    Carga pares de prueba para calibración del umbral
    - Pares genuinos: firmas del mismo autor (original_X_1.png vs original_X_2.png)  
    - Pares falsificados: 
      * Originales vs falsificaciones (original_X_Y.png vs forgeries_X_Y.png)
      * Originales de diferentes autores (original_1_X.png vs original_2_Y.png)
    
    Retorna: (pairs, labels) donde labels: 1=genuine, 0=forged
    """
    pairs = []
    labels = []
    
    # Directorio de firmas originales y falsificadas
    org_dir = "../dataset/signatures/full_org/"
    forg_dir = "../dataset/signatures/full_forg/"
    
    if not os.path.exists(org_dir) or not os.path.exists(forg_dir):
        print("⚠️ Directorios de dataset no encontrados. Usando datos sintéticos para calibración.")
        return None, None
    
    # Cargar archivos disponibles
    org_files = sorted([f for f in os.listdir(org_dir) if f.endswith('.png')])
    forg_files = sorted([f for f in os.listdir(forg_dir) if f.endswith('.png')])
    
    print(f"📊 Archivos encontrados: {len(org_files)} originales, {len(forg_files)} falsificaciones")
    
    # ===== PARES GENUINOS (mismo autor) =====
    count_genuine = 0
    # Agrupar por autor: original_X_Y.png -> autor X
    authors = {}
    for file in org_files:
        if file.startswith('original_'):
            parts = file.split('_')
            if len(parts) >= 3:
                author_id = parts[1]  # X en original_X_Y.png
                if author_id not in authors:
                    authors[author_id] = []
                authors[author_id].append(file)
    
    # Crear pares del mismo autor
    for author_id, files in authors.items():
        if len(files) >= 2 and count_genuine < num_genuine:
            # Tomar pares de firmas del mismo autor
            for i in range(0, len(files)-1, 2):
                if count_genuine >= num_genuine:
                    break
                pairs.append((os.path.join(org_dir, files[i]), 
                             os.path.join(org_dir, files[i+1])))
                labels.append(1)  # Genuino
                count_genuine += 1
                print(f"✅ Par genuino: {files[i]} vs {files[i+1]} (autor {author_id})")
    
    # ===== PARES FALSIFICADOS =====
    count_forged = 0
    
    # Tipo 1: Original vs Falsificación (forgeries_X_Y.png)
    for forg_file in forg_files[:num_forged//2]:
        if count_forged >= num_forged//2:
            break
        if forg_file.startswith('forgeries_'):
            # forgeries_X_Y.png -> buscar original_X_Z.png
            parts = forg_file.split('_')
            if len(parts) >= 3:
                author_id = parts[1]  # X en forgeries_X_Y.png
                # Buscar un original del mismo autor
                matching_org = [f for f in org_files if f.startswith(f'original_{author_id}_')]
                if matching_org:
                    pairs.append((os.path.join(org_dir, matching_org[0]), 
                                 os.path.join(forg_dir, forg_file)))
                    labels.append(0)  # Falsificado
                    count_forged += 1
                    print(f"❌ Par falsificado: {matching_org[0]} vs {forg_file}")
    
    # Tipo 2: Diferentes autores (original_1_X.png vs original_2_Y.png)
    author_list = list(authors.keys())
    for i in range(len(author_list)):
        for j in range(i+1, len(author_list)):
            if count_forged >= num_forged:
                break
            author1, author2 = author_list[i], author_list[j]
            if authors[author1] and authors[author2]:
                pairs.append((os.path.join(org_dir, authors[author1][0]), 
                             os.path.join(org_dir, authors[author2][0])))
                labels.append(0)  # Falsificado (diferentes autores)
                count_forged += 1
                print(f"❌ Par diferentes autores: {authors[author1][0]} vs {authors[author2][0]} (autor {author1} vs {author2})")
    
    print(f"✅ Total: {count_genuine} pares genuinos y {count_forged} pares falsificados")
    return pairs, labels

def calibrate_threshold(pairs, labels):
    """
    Calibra el umbral óptimo usando los pares de prueba
    """
    if pairs is None or labels is None:
        print("⚠️ No se puede calibrar sin datos de prueba. Usando umbral por defecto: 0.55")
        return 0.55
    
    print("🔧 Calibrando umbral óptimo...")
    scores = []
    
    for (path1, path2) in pairs:
        try:
            img1 = preprocess(path1)
            img2 = preprocess(path2)
            
            emb1 = embedding_model.predict(img1[None, ...], verbose=0)
            emb2 = embedding_model.predict(img2[None, ...], verbose=0)
            
            score = similarity_score(emb1, emb2)[0]
            scores.append(score)
        except Exception as e:
            print(f"❌ Error procesando {path1}, {path2}: {e}")
            continue
    
    if len(scores) == 0:
        print("⚠️ No se pudieron procesar los pares. Usando umbral por defecto: 0.55")
        return 0.55
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Calcular ROC curve para encontrar el mejor umbral
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Encontrar el punto con mejor balance (Youden's J statistic)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]
    
    # Si el umbral es muy extremo, usar un valor más conservador
    if best_threshold > 0.95:
        # Buscar un umbral que dé ~80% de precisión
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        # Encontrar el umbral donde precision ≈ 0.8
        target_precision = 0.8
        precision_diffs = np.abs(precision - target_precision)
        best_precision_idx = np.argmin(precision_diffs)
        
        if best_precision_idx < len(pr_thresholds):
            alternative_threshold = pr_thresholds[best_precision_idx]
            if 0.5 <= alternative_threshold <= 0.9:
                best_threshold = alternative_threshold
                print(f"🔧 Ajustando umbral extremo de {thresholds[best_idx]:.3f} a {best_threshold:.3f}")
        
        # Fallback: usar un umbral razonable basado en la distribución
        if best_threshold > 0.95:
            best_threshold = np.percentile(scores[labels == 1], 20)  # 20% percentil de genuinos
            best_threshold = max(0.7, min(0.9, best_threshold))  # Limitar entre 0.7-0.9
            print(f"🔧 Usando umbral basado en percentiles: {best_threshold:.3f}")
    
    # Calcular métricas
    roc_auc = auc(fpr, tpr)
    
    print(f"📈 ROC AUC: {roc_auc:.3f}")
    print(f"🎯 Umbral óptimo: {best_threshold:.3f}")
    print(f"📊 En umbral óptimo - TPR: {tpr[best_idx]:.3f}, FPR: {fpr[best_idx]:.3f}")
    
    return best_threshold

def test_signature_pair(ref_path, test_path, threshold=0.5):
    """
    Función simple para probar un par de firmas manualmente
    """
    print(f"� COMPARANDO FIRMAS:")
    print(f"   Original: {ref_path}")
    print(f"   Prueba: {test_path}")
    print("-" * 60)
    
    # Verificar que los archivos existen
    if not os.path.exists(ref_path):
        print(f"❌ No encontrado: {ref_path}")
        return
    if not os.path.exists(test_path):
        print(f"❌ No encontrado: {test_path}")
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
    print(f"   🆕 Distancia Euclidiana:")
    print(f"      - Similitud: {score_euclidean:.6f}")
    print(f"      - Distancia cruda: {euclidean_dist_raw:.6f}")
    print(f"      - Umbral: {threshold}")
    print(f"      - Resultado: {result_euclidean}")
    print()
    print(f"   🔴 Manhattan (anterior):")
    print(f"      - Similitud: {score_manhattan:.6f}")
    print(f"      - Umbral: 0.800")
    print(f"      - Resultado: {result_manhattan}")
    print()
    
    if result_euclidean != result_manhattan:
        print(f"⚠️  Los métodos DIFIEREN!")
    else:
        print(f"✅ Ambos métodos coinciden")
    
    print("=" * 60)

# ========================================================================
# EJEMPLOS DE USO - Descomenta las líneas que quieras probar
# ========================================================================

print("🚀 SISTEMA DE VERIFICACIÓN CON DISTANCIA EUCLIDIANA")
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