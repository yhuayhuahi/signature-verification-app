#!/usr/bin/env python3
"""
Script para probar la funcionalidad del endpoint /verify
"""

import requests
import os

# URL base del servidor Flask
BASE_URL = "http://localhost:5000"

def test_register_signature():
    """Prueba el endpoint de registro de firma"""
    print("🔄 Probando registro de firma...")
    
    # Crear una imagen de prueba (simulada)
    test_image_path = "test_signature.png"
    
    # Verificar si existe una imagen de prueba
    if not os.path.exists(test_image_path):
        print("⚠️  No se encontró imagen de prueba, usa una imagen real para probar")
        return False
    
    with open(test_image_path, 'rb') as f:
        files = {'file': f}
        data = {'user_id': 'test_user'}
        
        try:
            response = requests.post(f"{BASE_URL}/register", files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def test_verify_signature():
    """Prueba el endpoint de verificación de firma"""
    print("🔄 Probando verificación de firma...")
    
    test_image_path = "test_signature.png"
    
    if not os.path.exists(test_image_path):
        print("⚠️  No se encontró imagen de prueba, usa una imagen real para probar")
        return False
    
    with open(test_image_path, 'rb') as f:
        files = {'file': f}
        data = {'user_id': 'test_user'}
        
        try:
            response = requests.post(f"{BASE_URL}/verify", files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

if __name__ == "__main__":
    print("🎯 Script de prueba para verificador de firmas")
    print("📝 Asegúrate de que el servidor Flask esté ejecutándose en localhost:5000")
    print("📝 Coloca una imagen llamada 'test_signature.png' en este directorio para probar")
    
    # test_register_signature()
    # test_verify_signature()
