import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils import build_embedding_model, SIZE, load_signature_images, create_pairs

# Importar wandb
import wandb
from wandb.integration.keras import WandbCallback

# Definir callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Inicializar W&B con claves de Kaggle
wandb.login(key="727ac55a8c69f45358face1c830a80c83285e958")

# Inicializar wandb
run = wandb.init(project="Verificacion_firmas_06", name="Test01")

# Definir entradas para la Red Siamesa
input_a = tf.keras.layers.Input(shape=(SIZE, SIZE, 1), name='input1')
input_b = tf.keras.layers.Input(shape=(SIZE, SIZE, 1), name='input2')

# Modelo de embeddings compartido
embedding_model = build_embedding_model((SIZE, SIZE, 1))
em_one = embedding_model(input_a)
em_two = embedding_model(input_b)

# Usar diferencia absoluta en lugar de concatenación
out = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([em_one, em_two])

# Capas completamente conectadas para clasificación
out = tf.keras.layers.Dense(64, activation='relu')(out)
out = tf.keras.layers.Dense(1, activation='sigmoid', name='Output')(out)

# Crear y compilar modelo
model = tf.keras.models.Model([input_a, input_b], out)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Carga y preprocesamiento de datos
data_dir = "../dataset/signatures"
genuine_dir = os.path.join(data_dir, 'full_org')
forgery_dir = os.path.join(data_dir, 'full_forg')

# Cargar las imagenes de las firmas tanto genuinas como falsificadas
genuine_images, forgery_images = load_signature_images(genuine_dir, forgery_dir)
genuine_images, forgery_images = genuine_images / 255.0, forgery_images / 255.0  # Normalizar

# Crear pares positivos y negativos
pairs, labels = create_pairs(genuine_images, forgery_images)
X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)

# Entrenar el modelo
history = model.fit(
    [X_train[:, 0], X_train[:, 1]], y_train,
    validation_data = ([X_test[:, 0], X_test[:, 1]], y_test),
    epochs = 20, batch_size = 32,
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    ]
)

# Evaluar el modelo
loss, accuracy = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, verbose=0)
predictions = model.predict([X_test[:, 0], X_test[:, 1]])
predicted_labels = (predictions > 0.5).astype(int)

# Generar curva ROC
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)

# Curva ROC
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend()
wandb.log({"Curva ROC": wandb.Image(plt)})
plt.close()

# ========================================================================
# ANÁLISIS DE EMBEDDINGS Y VISUALIZACIONES AVANZADAS
# ========================================================================

def visualizar_embeddings_2d(embedding_model, X_test, y_test, method='tsne'):
    """
    Visualiza los embeddings en 2D usando t-SNE o PCA
    """
    print(f"\n🔍 Generando visualización de embeddings usando {method.upper()}...")
    
    # Obtener embeddings para un subconjunto de datos (para acelerar t-SNE)
    n_samples = min(500, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Generar embeddings para ambas imágenes del par
    embeddings_a = embedding_model.predict(X_test[indices, 0])
    embeddings_b = embedding_model.predict(X_test[indices, 1])
    
    # Combinar embeddings
    all_embeddings = np.vstack([embeddings_a, embeddings_b])
    
    # Crear etiquetas (0: imagen de par falsificado, 1: imagen de par genuino)
    labels_a = y_test[indices]
    labels_b = y_test[indices]
    all_labels = np.hstack([labels_a, labels_b])
    
    # Reducir dimensionalidad
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    # Crear el gráfico
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Colorear por tipo de firma
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue']
    labels_text = ['Falsificada', 'Genuina']
    
    for i, (color, label) in enumerate(zip(colors, labels_text)):
        mask = all_labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=color, alpha=0.6, s=30, label=label)
    
    plt.xlabel(f'{method.upper()} Componente 1')
    plt.ylabel(f'{method.upper()} Componente 2')
    plt.title(f'Embeddings de Firmas en Espacio 2D ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Densidad de puntos
    plt.subplot(1, 2, 2)
    plt.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=20, cmap='YlOrRd')
    plt.xlabel(f'{method.upper()} Componente 1')
    plt.ylabel(f'{method.upper()} Componente 2')
    plt.title('Densidad de Embeddings')
    plt.colorbar(label='Densidad')
    
    plt.tight_layout()
    wandb.log({f"Embeddings_{method.upper()}": wandb.Image(plt)})
    plt.close()
    
    return embeddings_2d, all_labels

def visualizar_distribucion_distancias(embedding_model, X_test, y_test):
    """
    Visualiza la distribución de distancias entre embeddings
    """
    print("\n📏 Analizando distribución de distancias entre embeddings...")
    
    # Calcular embeddings
    n_samples = min(300, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    embeddings_a = embedding_model.predict(X_test[indices, 0])
    embeddings_b = embedding_model.predict(X_test[indices, 1])
    
    # Calcular distancias
    distances_euclidean = np.linalg.norm(embeddings_a - embeddings_b, axis=1)
    distances_cosine = 1 - np.sum(embeddings_a * embeddings_b, axis=1) / (
        np.linalg.norm(embeddings_a, axis=1) * np.linalg.norm(embeddings_b, axis=1))
    
    labels = y_test[indices]
    
    # Crear visualización
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distancia Euclidiana
    axes[0, 0].hist(distances_euclidean[labels == 0], alpha=0.7, label='Falsificadas', 
                   color='red', bins=20, density=True)
    axes[0, 0].hist(distances_euclidean[labels == 1], alpha=0.7, label='Genuinas', 
                   color='blue', bins=20, density=True)
    axes[0, 0].set_xlabel('Distancia Euclidiana')
    axes[0, 0].set_ylabel('Densidad')
    axes[0, 0].set_title('Distribución de Distancias Euclidianas')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distancia Coseno
    axes[0, 1].hist(distances_cosine[labels == 0], alpha=0.7, label='Falsificadas', 
                   color='red', bins=20, density=True)
    axes[0, 1].hist(distances_cosine[labels == 1], alpha=0.7, label='Genuinas', 
                   color='blue', bins=20, density=True)
    axes[0, 1].set_xlabel('Distancia Coseno')
    axes[0, 1].set_ylabel('Densidad')
    axes[0, 1].set_title('Distribución de Distancias Coseno')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot de distancias
    data_euclidean = [distances_euclidean[labels == 0], distances_euclidean[labels == 1]]
    axes[1, 0].boxplot(data_euclidean, labels=['Falsificadas', 'Genuinas'])
    axes[1, 0].set_ylabel('Distancia Euclidiana')
    axes[1, 0].set_title('Comparación de Distancias Euclidianas')
    axes[1, 0].grid(True, alpha=0.3)
    
    data_cosine = [distances_cosine[labels == 0], distances_cosine[labels == 1]]
    axes[1, 1].boxplot(data_cosine, labels=['Falsificadas', 'Genuinas'])
    axes[1, 1].set_ylabel('Distancia Coseno')
    axes[1, 1].set_title('Comparación de Distancias Coseno')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"Distribución_Distancias": wandb.Image(plt)})
    plt.close()
    
    # Estadísticas
    print("\n📊 Estadísticas de Distancias:")
    print(f"Distancia Euclidiana - Genuinas: μ={np.mean(distances_euclidean[labels == 1]):.4f}, σ={np.std(distances_euclidean[labels == 1]):.4f}")
    print(f"Distancia Euclidiana - Falsificadas: μ={np.mean(distances_euclidean[labels == 0]):.4f}, σ={np.std(distances_euclidean[labels == 0]):.4f}")
    print(f"Distancia Coseno - Genuinas: μ={np.mean(distances_cosine[labels == 1]):.4f}, σ={np.std(distances_cosine[labels == 1]):.4f}")
    print(f"Distancia Coseno - Falsificadas: μ={np.mean(distances_cosine[labels == 0]):.4f}, σ={np.std(distances_cosine[labels == 0]):.4f}")
    
    return distances_euclidean, distances_cosine, labels

def visualizar_predicciones_vs_distancias(predictions, X_test, y_test, embedding_model):
    """
    Gráfico de dispersión: predicciones vs distancias reales
    """
    print("\nCreando gráfico de predicciones vs distancias...")
    
    # Calcular distancias reales
    n_samples = min(400, len(X_test))
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    embeddings_a = embedding_model.predict(X_test[indices, 0])
    embeddings_b = embedding_model.predict(X_test[indices, 1])
    distances = np.linalg.norm(embeddings_a - embeddings_b, axis=1)
    
    pred_subset = predictions[indices].flatten()
    labels_subset = y_test[indices]
    
    # Crear gráfico
    plt.figure(figsize=(12, 5))
    
    # Gráfico 1: Predicción vs Distancia
    plt.subplot(1, 2, 1)
    colors = ['red' if label == 0 else 'blue' for label in labels_subset]
    plt.scatter(distances, pred_subset, c=colors, alpha=0.6, s=30)
    plt.xlabel('Distancia Euclidiana entre Embeddings')
    plt.ylabel('Predicción del Modelo (Probabilidad)')
    plt.title('Predicción vs Distancia en Espacio de Embeddings')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Umbral (0.5)')
    plt.legend(['Umbral', 'Falsificadas', 'Genuinas'])
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Histograma 2D
    plt.subplot(1, 2, 2)
    plt.hist2d(distances, pred_subset, bins=20, cmap='Blues')
    plt.xlabel('Distancia Euclidiana')
    plt.ylabel('Predicción del Modelo')
    plt.title('Densidad de Predicciones vs Distancias')
    plt.colorbar(label='Frecuencia')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    wandb.log({"Predicciones_vs_Distancias": wandb.Image(plt)})
    plt.close()
    
    # Correlación
    correlation = np.corrcoef(distances, pred_subset)[0, 1]
    print(f"Correlación entre distancia y predicción: {correlation:.4f}")
    
    return correlation

def crear_mapa_calor_similitudes(embedding_model, X_test, y_test):
    """
    Crea un mapa de calor mostrando similitudes entre muestras
    """
    print("\n Generando mapa de calor de similitudes...")
    
    # Usar una muestra pequeña para el mapa de calor
    n_samples = 20
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Obtener embeddings
    embeddings_a = embedding_model.predict(X_test[indices, 0])
    embeddings_b = embedding_model.predict(X_test[indices, 1])
    
    # Combinar todos los embeddings
    all_embeddings = np.vstack([embeddings_a, embeddings_b])
    labels = np.hstack([y_test[indices], y_test[indices]])
    
    # Calcular matriz de similitudes (similitud coseno)
    similarity_matrix = np.dot(all_embeddings, all_embeddings.T) / (
        np.linalg.norm(all_embeddings, axis=1)[:, np.newaxis] * 
        np.linalg.norm(all_embeddings, axis=1)[np.newaxis, :]
    )
    
    # Crear etiquetas para el mapa
    label_names = ['F' if l == 0 else 'G' for l in labels]
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # Agregar etiquetas
    plt.xticks(range(len(label_names)), label_names)
    plt.yticks(range(len(label_names)), label_names)
    
    # Agregar valores en las celdas
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(similarity_matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, label='Similitud Coseno')
    plt.title('Mapa de Calor de Similitudes entre Embeddings\n(F=Falsificada, G=Genuina)')
    plt.xlabel('Muestras')
    plt.ylabel('Muestras')
    
    wandb.log({"Mapa_Calor_Similitudes": wandb.Image(plt)})
    plt.close()

# Ejecutar todas las visualizaciones
print("\n" + "="*60)
print("ANÁLISIS VISUAL DE EMBEDDINGS")
print("="*60)

# 1. Visualización de embeddings en 2D
embeddings_2d_tsne, labels_2d = visualizar_embeddings_2d(embedding_model, X_test, y_test, 'tsne')
embeddings_2d_pca, _ = visualizar_embeddings_2d(embedding_model, X_test, y_test, 'pca')

# 2. Distribución de distancias
dist_euclidean, dist_cosine, dist_labels = visualizar_distribucion_distancias(embedding_model, X_test, y_test)

# 3. Predicciones vs distancias
correlation = visualizar_predicciones_vs_distancias(predictions, X_test, y_test, embedding_model)

# 4. Mapa de calor de similitudes
crear_mapa_calor_similitudes(embedding_model, X_test, y_test)

# Registrar métricas adicionales en wandb
wandb.log({
    "Correlación_Distancia_Predicción": correlation,
    "Media_Distancia_Genuinas": np.mean(dist_euclidean[dist_labels == 1]),
    "Media_Distancia_Falsificadas": np.mean(dist_euclidean[dist_labels == 0]),
    "Separabilidad_Embeddings": np.mean(dist_euclidean[dist_labels == 0]) - np.mean(dist_euclidean[dist_labels == 1])
})

print(f"\n✅ Visualizaciones de embeddings completadas")
print(f"Separabilidad en espacio de embeddings: {np.mean(dist_euclidean[dist_labels == 0]) - np.mean(dist_euclidean[dist_labels == 1]):.4f}")

# Mostrar resultados 
print("\n" + "="*50)
print("RESULTADOS DE LA EVALUACIÓN")
print("="*50)
print(f"Precisión de Prueba: {accuracy:.4f}")
print(f"AUC (Área bajo la curva): {roc_auc:.3f}")

# Generar reporte de clasificación 
report_dict = classification_report(y_test, predicted_labels, output_dict=True)
print("\nReporte de Clasificación:")
print("-" * 60)
print(f"{'Clase':<15} {'Precisión':<12} {'Recall':<12} {'F1-Score':<12} {'Soporte':<10}")
print("-" * 60)
print(f"{'Falsificada':<15} {report_dict['0']['precision']:<12.3f} {report_dict['0']['recall']:<12.3f} {report_dict['0']['f1-score']:<12.3f} {report_dict['0']['support']:<10}")
print(f"{'Genuina':<15} {report_dict['1']['precision']:<12.3f} {report_dict['1']['recall']:<12.3f} {report_dict['1']['f1-score']:<12.3f} {report_dict['1']['support']:<10}")
print("-" * 60)
print(f"{'Promedio Macro':<15} {report_dict['macro avg']['precision']:<12.3f} {report_dict['macro avg']['recall']:<12.3f} {report_dict['macro avg']['f1-score']:<12.3f} {report_dict['macro avg']['support']:<10}")
print(f"{'Promedio Pesado':<15} {report_dict['weighted avg']['precision']:<12.3f} {report_dict['weighted avg']['recall']:<12.3f} {report_dict['weighted avg']['f1-score']:<12.3f} {report_dict['weighted avg']['support']:<10}")
print("-" * 60)

# Métricas adicionales 
total_samples = len(y_test)
correct_predictions = np.sum(y_test == predicted_labels.flatten())
false_positives = np.sum((y_test == 0) & (predicted_labels.flatten() == 1))
false_negatives = np.sum((y_test == 1) & (predicted_labels.flatten() == 0))
true_positives = np.sum((y_test == 1) & (predicted_labels.flatten() == 1))
true_negatives = np.sum((y_test == 0) & (predicted_labels.flatten() == 0))

print(f"\nMétricas Adicionales:")
print(f"Total de muestras: {total_samples}")
print(f"Predicciones correctas: {correct_predictions}")
print(f"Falsos positivos (firmas falsas clasificadas como genuinas): {false_positives}")
print(f"Falsos negativos (firmas genuinas clasificadas como falsas): {false_negatives}")
print(f"Verdaderos positivos (firmas genuinas correctamente identificadas): {true_positives}")
print(f"Verdaderos negativos (firmas falsas correctamente identificadas): {true_negatives}")
print("="*50)

# Registrar resultados en wandb
wandb.config.update({
    "tipo_modelo": "red_siamesa",
    "tamaño_imagen": 128,
    "tamaño_lote": 32,
    "tasa_aprendizaje": 0.01,
    "epocas": 20
})

# Registrar métricas de prueba
wandb.log({
    "Precisión_Prueba": accuracy,
    "Pérdida_Prueba": loss,
    "AUC": roc_auc,
})

# Registrar métricas de clasificación
report = classification_report(y_test, predicted_labels, output_dict=True)
wandb.log({
    "Precisión_Falsificada": report['0']['precision'],
    "Recall_Falsificada": report['0']['recall'],
    "F1_Falsificada": report['0']['f1-score'],
    "Precisión_Genuina": report['1']['precision'],
    "Recall_Genuina": report['1']['recall'],
    "F1_Genuina": report['1']['f1-score']
})

# Crear y registrar matriz de confusión
cm = confusion_matrix(y_test, predicted_labels)

# Extraer valores fundamentales de la matriz de confusión
# Para clasificación binaria: cm = [[TN, FP], [FN, TP]]
tn, fp, fn, tp = cm.ravel()

print(f"\n" + "="*60)
print("ANÁLISIS DETALLADO DE LA MATRIZ DE CONFUSIÓN")
print("="*60)
print("Valores fundamentales:")
print(f"• Verdaderos Negativos (TN): {tn}")
print(f"• Falsos Positivos (FP): {fp}")
print(f"• Falsos Negativos (FN): {fn}")
print(f"• Verdaderos Positivos (TP): {tp}")

print(f"\nInterpretación:")
print(f"• TN ({tn}): Firmas falsificadas correctamente identificadas como falsas")
print(f"• FP ({fp}): Firmas falsificadas incorrectamente identificadas como genuinas")
print(f"• FN ({fn}): Firmas genuinas incorrectamente identificadas como falsas")
print(f"• TP ({tp}): Firmas genuinas correctamente identificadas como genuinas")

# Calcular métricas usando las fórmulas fundamentales
precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precisión para clase 0 (falsificada)
recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0     # Recall para clase 0 (falsificada)
precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precisión para clase 1 (genuina)
recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0     # Recall para clase 1 (genuina)

# Métricas generales
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Especificidad
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensibilidad (igual a recall_1)

print(f"\n" + "="*60)
print("FÓRMULAS Y CÁLCULOS DE MÉTRICAS")
print("="*60)
print("📊 Métricas calculadas usando las fórmulas fundamentales:")
print(f"\n1. EXACTITUD (Accuracy):")
print(f"   Fórmula: (TP + TN) / (TP + TN + FP + FN)")
print(f"   Cálculo: ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {accuracy_manual:.4f}")

print(f"\n2. PRECISIÓN para Firmas Genuinas:")
print(f"   Fórmula: TP / (TP + FP)")
print(f"   Cálculo: {tp} / ({tp} + {fp}) = {precision_1:.4f}")

print(f"\n3. RECALL (Sensibilidad) para Firmas Genuinas:")
print(f"   Fórmula: TP / (TP + FN)")
print(f"   Cálculo: {tp} / ({tp} + {fn}) = {recall_1:.4f}")

print(f"\n4. ESPECIFICIDAD para Firmas Falsificadas:")
print(f"   Fórmula: TN / (TN + FP)")
print(f"   Cálculo: {tn} / ({tn} + {fp}) = {specificity:.4f}")

print(f"\n5. PRECISIÓN para Firmas Falsificadas:")
print(f"   Fórmula: TN / (TN + FN)")
print(f"   Cálculo: {tn} / ({tn} + {fn}) = {precision_0:.4f}")

print(f"\n6. RECALL para Firmas Falsificadas:")
print(f"   Fórmula: TN / (TN + FP)")
print(f"   Cálculo: {tn} / ({tn} + {fp}) = {recall_0:.4f}")

# F1-Scores calculados manualmente
f1_genuine = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
f1_forged = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

print(f"\n7. F1-SCORE para Firmas Genuinas:")
print(f"   Fórmula: 2 * (Precisión * Recall) / (Precisión + Recall)")
print(f"   Cálculo: 2 * ({precision_1:.4f} * {recall_1:.4f}) / ({precision_1:.4f} + {recall_1:.4f}) = {f1_genuine:.4f}")

print(f"\n8. F1-SCORE para Firmas Falsificadas:")
print(f"   Fórmula: 2 * (Precisión * Recall) / (Precisión + Recall)")
print(f"   Cálculo: 2 * ({precision_0:.4f} * {recall_0:.4f}) / ({precision_0:.4f} + {recall_0:.4f}) = {f1_forged:.4f}")

print(f"\n" + "="*60)
print("RESUMEN DE RENDIMIENTO DEL MODELO")
print("="*60)
print(f"🎯 Exactitud General: {accuracy_manual:.4f} ({accuracy_manual*100:.2f}%)")
print(f"🔍 Sensibilidad (detección de genuinas): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"🛡️  Especificidad (detección de falsas): {specificity:.4f} ({specificity*100:.2f}%)")
print(f"⚖️  Balance del modelo: {'Bueno' if abs(sensitivity - specificity) < 0.1 else 'Desbalanceado'}")

# Registrar valores en wandb
wandb.log({
    "TN_Verdaderos_Negativos": int(tn),
    "FP_Falsos_Positivos": int(fp),
    "FN_Falsos_Negativos": int(fn),
    "TP_Verdaderos_Positivos": int(tp),
    "Exactitud_Manual": accuracy_manual,
    "Sensibilidad": sensitivity,
    "Especificidad": specificity,
    "F1_Genuinas_Manual": f1_genuine,
    "F1_Falsificadas_Manual": f1_forged
})

# Versión en español
plt.figure(figsize=(8, 6))
disp_es = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Falsificada', 'Genuina'])
disp_es.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")

# Agregar anotaciones con los valores TN, FP, FN, TP
plt.text(0, 0, f'TN\n{tn}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
plt.text(1, 0, f'FP\n{fp}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
plt.text(0, 1, f'FN\n{fn}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
plt.text(1, 1, f'TP\n{tp}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

wandb.log({"Matriz de Confusión": wandb.Image(plt)})
plt.close()

# Registrar historial de entrenamiento
for epoch in range(len(history.history['loss'])):
    wandb.log({
        "Época": epoch,
        "Pérdida_Entrenamiento": history.history['loss'][epoch],
        "Pérdida_Validación": history.history['val_loss'][epoch],
        "Precisión_Entrenamiento": history.history['accuracy'][epoch],
        "Precisión_Validación": history.history['val_accuracy'][epoch],
    })

# Guardar el modelo
model_path = 'modelo_siamesa_firmas.keras'
model.save(model_path)

# Registrar el modelo en wandb como artefacto
artifact = wandb.Artifact('modelo_siamesa_firmas', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)

# Guardar el modelo de embeddings por separado
embedding_model_path = 'modelo_embeddings_firmas.keras'
embedding_model.save(embedding_model_path)
embed_artifact = wandb.Artifact('modelo_embeddings_firmas', type='model')
embed_artifact.add_file(embedding_model_path)
wandb.log_artifact(embed_artifact)

# Registrar finalización y cerrar wandb
wandb.log({"estado": "completado"})
wandb.finish()

print("¡Entrenamiento y evaluación completados exitosamente!")