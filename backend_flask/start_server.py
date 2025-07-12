#!/usr/bin/env python3
"""
Script mejorado para iniciar el servidor Flask con manejo de errores
"""

import os
import sys

def main():
    print("🚀 Iniciando servidor de verificación de firmas...")
    
    # Verificar que el modelo existe
    if not os.path.exists("modelo_siamesa.h5"):
        print("❌ Error: No se encontró el archivo modelo_siamesa.h5")
        print("📁 Asegúrate de que el modelo esté en el directorio actual")
        return False
    
    # Verificar que la base de datos se pueda crear/acceder
    try:
        import sqlite3
        conn = sqlite3.connect("signatures.db")
        conn.close()
        print("✅ Base de datos accesible")
    except Exception as e:
        print(f"❌ Error con la base de datos: {e}")
        return False
    
    # Verificar imports críticos
    try:
        from model_loader import predict_similarity
        from preprocess import preprocess_signature
        print("✅ Módulos críticos importados correctamente")
    except Exception as e:
        print(f"❌ Error importando módulos: {e}")
        return False
    
    # Iniciar la aplicación Flask
    try:
        from app import app, init_db
        
        # Inicializar la base de datos
        init_db()
        print("✅ Base de datos inicializada")
        
        print("\n🌐 Servidor iniciado en http://localhost:5000")
        print("📋 Endpoints disponibles:")
        print("   POST /register - Registrar firma original")
        print("   POST /verify   - Verificar firma")
        print("\n🔍 Para detener el servidor presiona Ctrl+C")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Error iniciando el servidor: {e}")
        return False

if __name__ == "__main__":
    main()
