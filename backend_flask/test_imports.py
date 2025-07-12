#!/usr/bin/env python3
"""
Script para probar que todos los imports funcionen correctamente.
"""

print("🔄 Probando imports...")

try:
    import numpy as np
    print("✅ numpy importado correctamente")
except Exception as e:
    print(f"❌ Error importando numpy: {e}")

try:
    import tensorflow as tf
    print("✅ tensorflow importado correctamente")
except Exception as e:
    print(f"❌ Error importando tensorflow: {e}")

try:
    import cv2
    print("✅ opencv importado correctamente")
except Exception as e:
    print(f"❌ Error importando opencv: {e}")

try:
    from PIL import Image
    print("✅ PIL importado correctamente")
except Exception as e:
    print(f"❌ Error importando PIL: {e}")

try:
    from flask import Flask
    print("✅ Flask importado correctamente")
except Exception as e:
    print(f"❌ Error importando Flask: {e}")

try:
    from preprocess import preprocess_signature
    print("✅ preprocess importado correctamente")
except Exception as e:
    print(f"❌ Error importando preprocess: {e}")

try:
    from model_loader import predict_similarity
    print("✅ model_loader importado correctamente")
except Exception as e:
    print(f"❌ Error importando model_loader: {e}")

print("\n🎯 Todos los imports completados")
