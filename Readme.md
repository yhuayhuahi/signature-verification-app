# Verificador de Firmas

## Contexto del modelo:

INTELIGENCIA ARTIFICIAL
│
├── Aprendizaje Automático
│   ├── Supervisado
│   │   ├── Clasificación
│   │   │   └── Clasificación Binaria (1 o 0)
│   │   │       └── Red Siamesa con CNN
│   └── No supervisado (no aplica)
└── Otros enfoques (no usados aquí)

## Tentativa de estructura del proyecto

verificador_firmas/
│
├── modelo_siamesa/
│   ├── train_model.py           # Código para entrenar el modelo
│   ├── preprocess.py            # Funciones de preprocesamiento
│   ├── utils.py                 # Funciones auxiliares
│   └── modelo_siamesa.h5        # Modelo entrenado (cuando esté listo)
│
├── backend_flask/
│   ├── app.py                   # Código principal del servidor Flask
│   └── model_loader.py          # Carga del modelo y predicción
│
├── dataset/
│   ├── firmas_usuario1/
│   ├── firmas_usuario2/
│   └── ...
│
├── docs/                        # Informes, gráficas y materiales del informe
│
└── requirements.txt

Para subir a google colab:
from google.colab import files
uploaded = files.upload()

import os
print("Archivos subidos:", list(uploaded.keys()))

import zipfile

zip_path = "verificador_firmas.zip"
extract_path = "/content/verificador_firmas"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"✔️ Carpeta descomprimida en: {extract_path}")

%cd /content/verificador_firmas/modelo_siamesa
!python train_model.py