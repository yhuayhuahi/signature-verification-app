import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity, 
  Alert, 
  ScrollView,
  Image,
  ActivityIndicator 
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

// Configuración del backend
const BACKEND_URL = 'http://s34dhz-ip-190-239-94-58.tunnelmole.net';

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  // Función para seleccionar imagen de la galería
  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        setSelectedImage(result.assets[0]);
        setUploadResult(null);
        console.log('✅ Imagen seleccionada:', result.assets[0].uri);
      }
    } catch (error) {
      console.error('❌ Error seleccionando imagen:', error);
      Alert.alert('Error', 'No se pudo seleccionar la imagen');
    }
  };

  // Función para tomar foto con la cámara (envío automático)
  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permisos', 'Se necesita permiso para usar la cámara');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        const photo = result.assets[0];
        setSelectedImage(photo);
        setUploadResult(null);
        console.log('✅ Foto tomada:', photo.uri);
        
        // Enviar automáticamente después de tomar la foto
        console.log('🚀 Enviando foto automáticamente...');
        await uploadImageDirect(photo);
      }
    } catch (error) {
      console.error('❌ Error tomando foto:', error);
      Alert.alert('Error', 'No se pudo tomar la foto');
    }
  };

  // Función para enviar imagen directamente (sin usar el estado)
  const uploadImageDirect = async (imageAsset) => {
    setIsUploading(true);
    setUploadResult(null);

    try {
      console.log('🚀 Iniciando upload directo...');
      console.log('📁 URI:', imageAsset.uri);

      // Crear FormData
      const formData = new FormData();
      formData.append('file', {
        uri: imageAsset.uri,
        type: imageAsset.mimeType || 'image/jpeg',
        name: imageAsset.fileName || 'camera-photo.jpg',
      });
      formData.append('test_param', 'foto_automatica');
      formData.append('source', 'camera');

      console.log('📤 Enviando al backend...');

      // Enviar al backend
      const response = await fetch(`${BACKEND_URL}/test-image`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('📥 Status de respuesta:', response.status);
      
      const result = await response.json();
      console.log('📥 Respuesta del backend:', result);

      setUploadResult({
        success: response.ok,
        status: response.status,
        data: result,
        autoSent: true
      });

      if (response.ok) {
        Alert.alert('¡Éxito!', '✅ Foto enviada automáticamente al backend');
      } else {
        Alert.alert('Error', `❌ Error del servidor: ${result.error || 'Desconocido'}`);
      }

    } catch (error) {
      console.error('❌ Error en upload directo:', error);
      setUploadResult({
        success: false,
        error: error.message,
        autoSent: true
      });
      Alert.alert('Error', `❌ Error de conexión: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Función para enviar imagen desde galería (manual)
  const uploadImage = async () => {
    if (!selectedImage) {
      Alert.alert('Error', 'Primero selecciona una imagen');
      return;
    }

    setIsUploading(true);
    setUploadResult(null);

    try {
      console.log('🚀 Iniciando upload manual...');
      console.log('📁 URI:', selectedImage.uri);

      // Crear FormData
      const formData = new FormData();
      formData.append('file', {
        uri: selectedImage.uri,
        type: selectedImage.mimeType || 'image/jpeg',
        name: selectedImage.fileName || 'gallery-image.jpg',
      });
      formData.append('test_param', 'seleccion_manual');
      formData.append('source', 'gallery');

      console.log('📤 Enviando al backend...');

      // Enviar al backend
      const response = await fetch(`${BACKEND_URL}/test-image`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('📥 Status de respuesta:', response.status);
      
      const result = await response.json();
      console.log('📥 Respuesta del backend:', result);

      setUploadResult({
        success: response.ok,
        status: response.status,
        data: result,
        autoSent: false
      });

      if (response.ok) {
        Alert.alert('¡Éxito!', '✅ Imagen enviada correctamente al backend');
      } else {
        Alert.alert('Error', `❌ Error del servidor: ${result.error || 'Desconocido'}`);
      }

    } catch (error) {
      console.error('❌ Error en upload:', error);
      setUploadResult({
        success: false,
        error: error.message,
        autoSent: false
      });
      Alert.alert('Error', `❌ Error de conexión: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🧪 Test de Envío de Imágenes</Text>
        <Text style={styles.subtitle}>Backend: {BACKEND_URL}</Text>
      </View>

      {/* Botones para seleccionar imagen */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={pickImage}>
          <Text style={styles.buttonText}>📁 Seleccionar de Galería</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.button, styles.cameraButton, isUploading && styles.buttonDisabled]} 
          onPress={takePhoto}
          disabled={isUploading}
        >
          {isUploading ? (
            <View style={styles.buttonContent}>
              <ActivityIndicator color="white" size="small" style={{marginRight: 8}} />
              <Text style={styles.buttonText}>📷 Enviando...</Text>
            </View>
          ) : (
            <Text style={styles.buttonText}>📷 Tomar y Enviar Foto</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Mostrar estado de carga cuando se está enviando */}
      {isUploading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2196F3" />
          <Text style={styles.loadingText}>📤 Enviando foto al servidor...</Text>
        </View>
      )}

      {/* Mostrar imagen seleccionada desde galería */}
      {selectedImage && !isUploading && !uploadResult?.autoSent && (
        <View style={styles.imageContainer}>
          <Text style={styles.sectionTitle}>✅ Imagen Seleccionada:</Text>
          <Image source={{ uri: selectedImage.uri }} style={styles.image} />
          <Text style={styles.imageInfo}>
            📊 Tamaño: {selectedImage.width}x{selectedImage.height}
          </Text>
          <Text style={styles.imageInfo}>
            📁 Archivo: {selectedImage.fileName || 'Sin nombre'}
          </Text>
        </View>
      )}

      {/* Botón de upload manual para imágenes de galería */}
      {selectedImage && !isUploading && !uploadResult?.autoSent && (
        <TouchableOpacity 
          style={[styles.uploadButton, isUploading && styles.uploadButtonDisabled]} 
          onPress={uploadImage}
          disabled={isUploading}
        >
          <Text style={styles.uploadButtonText}>🚀 Enviar al Backend</Text>
        </TouchableOpacity>
      )}

      {/* Mostrar resultado */}
      {uploadResult && (
        <View style={styles.resultContainer}>
          <Text style={styles.sectionTitle}>
            {uploadResult.success ? '✅ Resultado Exitoso:' : '❌ Error:'}
          </Text>
          <Text style={styles.resultSubtitle}>
            {uploadResult.autoSent ? '📷 Enviado automáticamente desde cámara' : '📁 Enviado desde galería'}
          </Text>
          <Text style={styles.resultText}>
            {JSON.stringify(uploadResult, null, 2)}
          </Text>
        </View>
      )}

      <StatusBar style="auto" />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#2196F3',
    padding: 20,
    paddingTop: 50,
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 12,
    color: '#E3F2FD',
  },
  buttonContainer: {
    padding: 20,
    gap: 10,
  },
  button: {
    backgroundColor: '#4CAF50',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  cameraButton: {
    backgroundColor: '#FF9800',
  },
  buttonDisabled: {
    backgroundColor: '#BDBDBD',
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  loadingContainer: {
    margin: 20,
    padding: 20,
    backgroundColor: 'white',
    borderRadius: 8,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#333',
    fontWeight: 'bold',
  },
  imageContainer: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  resultSubtitle: {
    fontSize: 12,
    color: '#666',
    marginBottom: 10,
    fontStyle: 'italic',
  },
  image: {
    width: 200,
    height: 150,
    borderRadius: 8,
    marginBottom: 10,
  },
  imageInfo: {
    fontSize: 12,
    color: '#666',
    marginBottom: 2,
  },
  uploadButton: {
    backgroundColor: '#2196F3',
    margin: 20,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  uploadButtonDisabled: {
    backgroundColor: '#BDBDBD',
  },
  uploadButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultContainer: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultText: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#333',
    backgroundColor: '#f9f9f9',
    padding: 10,
    borderRadius: 4,
  },
});
