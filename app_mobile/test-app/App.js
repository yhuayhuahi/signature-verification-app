import React, { useState, useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, View, Alert, ScrollView } from 'react-native';
import {
  Header,
  ImagePickerButtons,
  ImagePreview,
  ActionButtons,
  UserIdSection,
  ResultDisplay,
  UserStatistics,
  useImageHandler
} from './components';
import * as apiService from './services/api';

// Configuración del backend
const DEFAULT_BACKEND_URL = 'http://s34dhz-ip-190-239-94-58.tunnelmole.net';

export default function App() {
  // Estados principales
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [userId, setUserId] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [verificationResult, setVerificationResult] = useState(null);
  const [stats, setStats] = useState(null);
  
  // Hook personalizado para manejo de imágenes
  const {
    selectedImage,
    isLoading: isImageLoading,
    pickImageFromGallery,
    takePhotoWithCamera,
    clearImage
  } = useImageHandler();

  // Cargar estadísticas cuando cambie el userId
  useEffect(() => {
    if (userId.trim()) {
      loadStats(userId);
    } else {
      setStats(null);
    }
  }, [userId]);

  // Manejar cambio de URL del backend
  const handleBackendUrlChange = (newUrl) => {
    setBackendUrl(newUrl);
    apiService.updateBackendUrl(newUrl);
    
    // Recargar estadísticas si hay un usuario activo
    if (userId.trim()) {
      loadStats(userId);
    }
  };

  // Función para registrar firma
  const registerSignature = async (image, userId) => {
    setIsProcessing(true);
    try {
      console.log('🚀 Registrando firma...');
      console.log('📁 URI:', image.uri);
      console.log('🆔 Usuario:', userId);
      
      // Llamada real al backend
      const response = await apiService.registerSignature(image, userId);
      
      console.log('✅ Respuesta del servidor:', response.data);
      
      Alert.alert('✅ Éxito', `Firma registrada para usuario: ${userId}`);
      clearImage();
      
      // Refrescar estadísticas
      await loadStats(userId);
      
    } catch (error) {
      console.error('❌ Error registrando firma:', error);
      
      let errorMessage = '❌ Error desconocido';
      if (error.response) {
        // Error de respuesta del servidor
        errorMessage = `❌ Error del servidor: ${error.response.data?.error || error.response.status}`;
      } else if (error.request) {
        // Error de red
        errorMessage = '❌ Error de conexión. Verifica tu red.';
      } else {
        // Error de configuración
        errorMessage = `❌ Error: ${error.message}`;
      }
      
      Alert.alert('Error', errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  // Función para verificar firma
  const verifySignature = async (image, userId) => {
    setIsProcessing(true);
    setVerificationResult(null);
    
    try {
      console.log('🔍 Verificando firma...');
      console.log('📁 URI:', image.uri);
      console.log('🆔 Usuario objetivo:', userId);
      
      // Llamada real al backend
      const response = await apiService.verifySignature(image, userId);
      
      console.log('✅ Respuesta del servidor:', response.data);
      
      // Procesar el resultado según la respuesta del backend
      const result = {
        userId: userId,
        similarity: response.data.max_similarity || response.data.similarity_score,
        isMatch: response.data.result === 'genuina',
        timestamp: new Date().toISOString(),
        totalReferences: response.data.total_references,
        method: response.data.method,
        threshold: response.data.threshold_used
      };
      
      setVerificationResult(result);
      
      if (result.isMatch) {
        Alert.alert('✅ Firma Válida', `Similitud: ${(result.similarity * 100).toFixed(1)}%`);
      } else {
        Alert.alert('❌ Firma No Válida', `Similitud: ${(result.similarity * 100).toFixed(1)}%`);
      }
      
      // Refrescar estadísticas
      await loadStats(userId);
      
    } catch (error) {
      console.error('❌ Error verificando firma:', error);
      
      let errorMessage = '❌ Error desconocido';
      if (error.response) {
        // Error de respuesta del servidor
        const serverError = error.response.data?.error || `Status: ${error.response.status}`;
        errorMessage = `❌ Error del servidor: ${serverError}`;
      } else if (error.request) {
        // Error de red
        errorMessage = '❌ Error de conexión. Verifica tu red.';
      } else {
        // Error de configuración
        errorMessage = `❌ Error: ${error.message}`;
      }
      
      Alert.alert('Error', errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  // Función para cargar estadísticas
  const loadStats = async (userIdParam) => {
    try {
      console.log('📊 Cargando estadísticas para:', userIdParam);
      
      // Llamada real al backend
      const response = await apiService.getUserStats(userIdParam);
      
      console.log('✅ Estadísticas del servidor:', response.data);
      
      setStats(response.data);
      
    } catch (error) {
      console.error('❌ Error cargando estadísticas:', error);
      
      // Usar datos de fallback en caso de error
      const fallbackStats = {
        user_id: userIdParam,
        registered_signatures: 0,
        message: "Error conectando al servidor. Mostrando datos locales."
      };
      
      setStats(fallbackStats);
      
      // No mostrar alert para estadísticas, solo log
      console.warn('⚠️ Usando datos de fallback para estadísticas');
    }
  };

  // Manejar selección de imagen desde galería
  const handlePickImage = async () => {
    setVerificationResult(null);
    await pickImageFromGallery();
  };

  // Manejar toma de foto
  const handleTakePhoto = async () => {
    setVerificationResult(null);
    await takePhotoWithCamera();
  };

  // Manejar registro directo
  const handleRegister = async (image, targetUserId) => {
    await registerSignature(image, targetUserId);
  };

  // Manejar verificación directa
  const handleVerify = async (image, targetUserId) => {
    await verifySignature(image, targetUserId);
  };

  const isLoading = isImageLoading || isProcessing;

  return (
    <View style={styles.container}>
      <Header 
        backendUrl={backendUrl} 
        onBackendUrlChange={handleBackendUrlChange}
      />
      
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Campo de ID de Usuario */}
        <UserIdSection
          userId={userId}
          onUserIdChange={setUserId}
        />

        {/* Botones de selección de imagen */}
        <ImagePickerButtons
          onPickImage={handlePickImage}
          onTakePhoto={handleTakePhoto}
          isUploading={isLoading}
        />

        {/* Vista previa de la imagen */}
        {selectedImage && (
          <ImagePreview image={selectedImage} />
        )}

        {/* Botones de acción */}
        <ActionButtons
          selectedImage={selectedImage}
          userId={userId}
          onRegisterSignature={handleRegister}
          onVerifySignature={handleVerify}
          isLoading={isLoading}
        />

        {/* Resultado de verificación */}
        {verificationResult && (
          <ResultDisplay result={verificationResult} />
        )}

        {/* Estadísticas del usuario */}
        <UserStatistics
          userId={userId}
          stats={stats}
          onRefreshStats={loadStats}
          isLoading={isLoading}
        />
      </ScrollView>
      
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 30, // Espacio extra al final
  },
});
