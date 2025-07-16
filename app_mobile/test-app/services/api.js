import axios from 'axios';

// Configuración base de la API
let BACKEND_URL = 'http://s34dhz-ip-190-239-94-58.tunnelmole.net';

// Crear instancia de API
let api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 30000, // 30 segundos
  headers: {
    'Accept': 'application/json',
  },
});

// Función para actualizar la URL del backend
export const updateBackendUrl = (newUrl) => {
  BACKEND_URL = newUrl;
  
  // Recrear la instancia de axios con la nueva URL
  api = axios.create({
    baseURL: BACKEND_URL,
    timeout: 30000,
    headers: {
      'Accept': 'application/json',
    },
  });

  // Reaplicar interceptors
  setupInterceptors();
  
  console.log('🔄 Backend URL actualizada a:', newUrl);
};

// Función para configurar interceptors
const setupInterceptors = () => {
  // Interceptor para logging
  api.interceptors.request.use(
    (config) => {
      console.log('🚀 API Request:', config.method?.toUpperCase(), config.url);
      return config;
    },
    (error) => {
      console.error('❌ API Request Error:', error);
      return Promise.reject(error);
    }
  );

  api.interceptors.response.use(
    (response) => {
      console.log('✅ API Response:', response.status, response.config.url);
      return response;
    },
    (error) => {
      console.error('❌ API Response Error:', error.response?.status, error.config?.url);
      return Promise.reject(error);
    }
  );
};

// Configurar interceptors inicialmente
setupInterceptors();

// Función para registrar firma
export const registerSignature = async (file, userId) => {
  const formData = new FormData();
  
  // Preparar el archivo según el formato esperado por React Native
  formData.append('file', {
    uri: file.uri,
    name: `${userId}_signature.jpg`,
    type: file.mimeType || 'image/jpeg',
  });
  
  formData.append('user_id', userId);

  const response = await api.post('/register', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response;
};

// Función para verificar firma
export const verifySignature = async (file, userId) => {
  const formData = new FormData();
  
  // Preparar el archivo según el formato esperado por React Native
  formData.append('file', {
    uri: file.uri,
    name: 'verification_signature.jpg',
    type: file.mimeType || 'image/jpeg',
  });
  
  formData.append('user_id', userId);

  const response = await api.post('/verify', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response;
};

// Función para obtener estadísticas de usuario
export const getUserStats = async (userId) => {
  const response = await api.get(`/stats?user_id=${userId}`);
  return response;
};

// Función de prueba de conectividad
export const testConnection = async () => {
  const response = await api.get('/');
  return response;
};

// Función de prueba para envío de imágenes (usando el endpoint test-image)
export const testImageUpload = async (file) => {
  const formData = new FormData();
  
  formData.append('file', {
    uri: file.uri,
    name: 'test.jpg',
    type: file.mimeType || 'image/jpeg',
  });

  const response = await api.post('/test-image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response;
};

export default api;
