import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

from preprocess import preprocess_signature
import random
from itertools import combinations

IMG_HEIGHT = 155
IMG_WIDTH = 220
IMG_CHANNELS = 1


def build_signature_encoder(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (10, 10), activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)

    return Model(inputs, x)


def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def load_signature_pairs(dataset_path, num_users=55, max_genuine_per_user=5):
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
            for _ in range(2):  # 2 negativos por cada genuina (ajustable)
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

