from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sqlite3
import uuid
import numpy as np

from model_loader import predict_similarity as load_model_and_predict
from preprocess import preprocess_signature

app = Flask(__name__)

# Configurar CORS para permitir solicitudes desde cualquier origen
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuraciones
UPLOAD_FOLDER = "uploads"
DB_PATH = "signatures.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicializar DB (solo si no existe)
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS firmas (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        filepath TEXT
                    )''')
        conn.commit()

@app.route("/")
def test_server():
    return jsonify({"message": "Servidor Flask funcionando correctamente"}), 200

# Ruta para probar CORS
@app.route("/test-cors", methods=["GET", "POST", "OPTIONS"])
def test_cors():
    return jsonify({
        "message": "CORS configurado correctamente",
        "method": request.method,
        "headers": dict(request.headers)
    }), 200

# Ruta para registrar firma original
@app.route("/register", methods=["POST"])
def register_signature():
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if not user_id or not file:
        return jsonify({"error": "user_id y file son requeridos"}), 400

    filename = secure_filename(file.filename)
    user_folder = os.path.join(UPLOAD_FOLDER, user_id)
    os.makedirs(user_folder, exist_ok=True)

    filepath = os.path.join(user_folder, f"{uuid.uuid4().hex}_{filename}")
    file.save(filepath)

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO firmas (user_id, filepath) VALUES (?, ?)", (user_id, filepath))
        conn.commit()

    return jsonify({"message": "Firma registrada exitosamente."})

# Ruta para verificar una firma
@app.route("/verify", methods=["POST"])
def verify_signature():
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if not user_id or not file:
        return jsonify({"error": "user_id y file son requeridos"}), 400

    try:
        # Preprocesar la imagen a verificar
        img_to_verify = preprocess_signature(file.stream)
    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 400

    # Obtener firmas originales del usuario
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT filepath FROM firmas WHERE user_id = ?", (user_id,))
        rows = c.fetchall()

    if not rows:
        return jsonify({"error": "No hay firmas originales registradas para este usuario."}), 404

    similarities = []
    for (path,) in rows:
        try:
            # Preprocesar la imagen de referencia desde el archivo
            img_ref = preprocess_signature(path)
            # Comparar las imágenes usando el modelo
            sim = load_model_and_predict(img_ref, img_to_verify)
            similarities.append(sim)
            print(f"✅ Similitud con {path}: {sim:.4f}")
        except Exception as e:
            print(f"❌ Error procesando {path}: {e}")
            continue

    if not similarities:
        return jsonify({"error": "Error procesando las firmas originales."}), 500

    avg_similarity = float(np.mean(similarities))
    max_similarity = float(np.max(similarities))
    is_genuine = avg_similarity > 0.5  # este umbral es ajustable

    return jsonify({
        "similarity_score": avg_similarity,
        "max_similarity": max_similarity,
        "similarities": similarities,
        "result": "genuina" if is_genuine else "falsa",
        "total_references": len(similarities)
    })

# Ruta para obtener estadísticas del usuario
@app.route("/stats", methods=["GET"])
def get_user_stats():
    user_id = request.args.get("user_id")
    
    if not user_id:
        return jsonify({"error": "user_id es requerido"}), 400
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            # Contar firmas registradas
            c.execute("SELECT COUNT(*) FROM firmas WHERE user_id = ?", (user_id,))
            registered_count = c.fetchone()[0]
            
            return jsonify({
                "user_id": user_id,
                "registered_signatures": registered_count,
                "message": "Estadísticas obtenidas exitosamente"
            })
    except Exception as e:
        return jsonify({"error": f"Error obteniendo estadísticas: {str(e)}"}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
