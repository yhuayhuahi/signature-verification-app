import tensorflow as tf

import cv2
import os
import numpy as np

SIZE = 128

def build_embedding_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(4, 4),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(4, 4),

        tf.keras.layers.Flatten(),
    ], name='embedding_model')
    return model

def load_signature_images(genuine_path, forgery_path, target_size=(128, 128)):
    def load_images(path):
        images = []
        for image_file in os.listdir(path):
            img = cv2.imread(os.path.join(path, image_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, target_size))
        return np.array(images)
    return load_images(genuine_path), load_images(forgery_path)

# Create Pairs for Siamese Training
def create_pairs(genuine, forged):
    pairs, labels = [], []
    for i in range(min(len(genuine), len(forged))):
        pairs.append([genuine[i], genuine[(i + 1) % len(genuine)]])  # Genuine pair
        labels.append(1)
        pairs.append([genuine[i], forged[i]])  # Forged pair
        labels.append(0)
    return np.array(pairs), np.array(labels)

'''
def load_signature_pairs(dataset_path, num_users=55, max_genuine_per_user=10):
    org_path = os.path.join(dataset_path, 'signatures', 'full_org')

    genuine_dict = {}

    # Agrupar firmas genuinas por usuario (ordenadas por nombre de archivo)
    for fname in sorted(os.listdir(org_path)):
        if fname.endswith(".png") or fname.endswith(".org"):
            parts = fname.split('_')
            if len(parts) < 3:
                continue
            user = int(parts[1])
            genuine_dict.setdefault(user, []).append(os.path.join(org_path, fname))

    # Seleccionamos los primeros N usuarios disponibles (ordenados)
    usuarios = sorted(genuine_dict.keys())[:num_users]
    X1, X2, y = [], [], []

    for user in usuarios:
        genuines = sorted(genuine_dict[user])[:max_genuine_per_user]  # Tomamos solo N muestras por usuario

        # Generar pares positivos (todas las combinaciones posibles entre genuinas de ese usuario)
        pos_pairs = list(combinations(genuines, 2))  # C(n,2)
        for img_path1, img_path2 in pos_pairs:
            img1 = preprocess_signature(img_path1)
            img2 = preprocess_signature(img_path2)
            X1.append(img1)
            X2.append(img2)
            y.append(1)

        # Generar pares negativos contra otros usuarios
        other_users = [u for u in usuarios if u != user]
        for img_path1 in genuines:
            for _ in range(3):  # 2 negativos por cada genuina (ajustable)
                neg_user = random.choice(other_users)
                neg_genuine = sorted(genuine_dict[neg_user])[:max_genuine_per_user]
                img_path2 = random.choice(neg_genuine)
                img1 = preprocess_signature(img_path1)
                img2 = preprocess_signature(img_path2)
                X1.append(img1)
                X2.append(img2)
                y.append(0)

    # Convertimos a arrays finales
    X1 = np.array(X1, dtype=np.float32)
    X2 = np.array(X2, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X1, X2, y
'''
