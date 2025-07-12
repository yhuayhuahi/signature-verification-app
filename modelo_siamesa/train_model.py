import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocess import preprocess_signature
from utils import load_signature_pairs, euclidean_distance
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

IMG_HEIGHT = 155
IMG_WIDTH = 220
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Define el encoder de firmas
from utils import build_signature_encoder

def build_siamese_network(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    encoder = build_signature_encoder(input_shape)
    encoded_a = encoder(input_a)
    encoded_b = encoder(input_b)

    distance = Lambda(euclidean_distance, name="euclidean_distance")([encoded_a, encoded_b])
    output = Dense(1, activation='sigmoid')(distance)

    model = Model(inputs=[input_a, input_b], outputs=output, name="SiameseNetwork")
    return model

def main():
    print("Cargando y preprocesando datos...")
    X1, X2, y = load_signature_pairs("../dataset")  # asume estructura con carpetas por usuario

    print("Construyendo el modelo...")
    model = build_siamese_network(INPUT_SHAPE)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("modelo_siamesa.h5", monitor='val_loss', save_best_only=True)

    print("Entrenando el modelo...")
    model.fit(
        [X1, X2], y,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        callbacks=[checkpoint]
    )

    print("Modelo entrenado y guardado como 'modelo_siamesa.h5'")

if __name__ == "__main__":
    main()
