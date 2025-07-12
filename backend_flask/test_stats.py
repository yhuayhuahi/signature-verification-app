#!/usr/bin/env python3
"""
Script para probar la nueva ruta de estadísticas
"""

import requests
import sys

# URL base del servidor Flask
BASE_URL = "http://localhost:5000"

def test_stats_endpoint():
    """Prueba el endpoint de estadísticas"""
    print("🔄 Probando endpoint de estadísticas...")
    
    # Usar un user_id de ejemplo
    test_user_id = "user_1234567890_test123"
    
    try:
        response = requests.get(f"{BASE_URL}/stats", params={'user_id': test_user_id})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Endpoint de estadísticas funciona correctamente")
            return True
        else:
            print("❌ Error en el endpoint")
            return False
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print("💡 Asegúrate de que el servidor Flask esté ejecutándose")
        return False

def test_all_endpoints():
    """Prueba todos los endpoints disponibles"""
    print("🎯 Probando todos los endpoints del API...")
    
    endpoints = [
        ("/register", "POST"),
        ("/verify", "POST"), 
        ("/stats", "GET")
    ]
    
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", params={'user_id': 'test'})
            else:
                response = requests.head(f"{BASE_URL}{endpoint}")
            
            print(f"📡 {method} {endpoint}: Status {response.status_code}")
            
        except Exception as e:
            print(f"❌ {method} {endpoint}: Error de conexión")

if __name__ == "__main__":
    print("🚀 Script de prueba para API de verificación de firmas")
    print("=" * 50)
    
    test_stats_endpoint()
    print()
    test_all_endpoints()
    
    print("\n✨ Pruebas completadas")
