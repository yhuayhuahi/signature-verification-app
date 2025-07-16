import cv2, numpy as np, tensorflow as tf, matplotlib.pyplot as plt

# ---------  carga del modelo de embedding (sin Lambda) -------------
embedding_model = tf.keras.models.load_model(
    "signature_embedding_model.keras", compile=False)

SIZE = 128        # o el que usaste en entrenamiento
THRESHOLD = 0.80  # ajusta si es necesario

def preprocess(path, size=SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size)).astype("float32") / 255.0
    return img[..., None]                     # (H,W,1)

def similarity_score(emb1, emb2):
    """Manhattan‑based similarity  ∈ [0,1]. 1 = idéntico"""
    return 1 - np.sum(np.abs(emb1 - emb2)) / np.sum(np.abs(emb1) + np.abs(emb2))

def verify_signature(ref_path, test_path, threshold=THRESHOLD):
    ref_img  = preprocess(ref_path)
    test_img = preprocess(test_path)

    # Visualización opcional
    for title, img, pos in [("Original", ref_path, 1), ("Compared", test_path, 2)]:
        plt.subplot(1,2,pos); plt.imshow(cv2.imread(img,0), cmap='gray')
        plt.title(title); plt.axis("off")
    plt.show()

    # ---------- inferencia ----------
    ref_emb  = embedding_model.predict(ref_img[None, ...], verbose=0)
    test_emb = embedding_model.predict(test_img[None, ...], verbose=0)
    score = similarity_score(ref_emb, test_emb)
    result = "Genuine" if score > threshold else "Forged"
    
    print(f"Comparing signatures:\n")
    print("Original Signature:", ref_path)
    print("Compared Signature:", test_path)

    print(f"Result: {result}   |   similarity = {score:.4f}")

# ----- Prueba -----
verify_signature(
    "../dataset/signatures/full_org/original_2_1.png",
    "../dataset/signatures/full_org/original_2_2.png"
)
