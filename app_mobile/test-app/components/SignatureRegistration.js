import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  Alert,
  TouchableOpacity 
} from 'react-native';
import { ImagePreview, ActionButton, LoadingIndicator } from './';

const SignatureRegistration = ({ 
  selectedImage, 
  onRegisterSignature, 
  isLoading 
}) => {
  const [userId, setUserId] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);

  // Generar ID aleatorio
  const generateRandomId = () => {
    const randomId = Math.random().toString(36).substring(2, 8).toUpperCase();
    setUserId(randomId);
  };

  const handleRegister = async () => {
    if (!selectedImage) {
      Alert.alert('Error', 'Primero selecciona una imagen de firma');
      return;
    }

    if (!userId.trim()) {
      Alert.alert('Error', 'Ingresa un ID para la firma');
      return;
    }

    setIsRegistering(true);
    try {
      await onRegisterSignature(selectedImage, userId.trim());
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>📝 Registrar Nueva Firma</Text>
      
      <ImagePreview image={selectedImage} />
      
      {selectedImage && (
        <View style={styles.formContainer}>
          <Text style={styles.label}>🆔 ID de Usuario:</Text>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.textInput}
              value={userId}
              onChangeText={setUserId}
              placeholder="Ingresa ID personalizado"
              maxLength={20}
              autoCapitalize="characters"
            />
            <TouchableOpacity 
              style={styles.generateButton}
              onPress={generateRandomId}
            >
              <Text style={styles.generateButtonText}>🎲 Generar</Text>
            </TouchableOpacity>
          </View>
          
          {isRegistering && (
            <LoadingIndicator message="Registrando firma..." />
          )}
          
          <ActionButton
            title="💾 Registrar Firma"
            onPress={handleRegister}
            isLoading={isRegistering}
            disabled={isRegistering || isLoading}
          />
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#333',
    marginBottom: 20,
  },
  formContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  textInput: {
    flex: 1,
    borderWidth: 2,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
    marginRight: 10,
  },
  generateButton: {
    backgroundColor: '#9C27B0',
    paddingHorizontal: 15,
    paddingVertical: 12,
    borderRadius: 8,
  },
  generateButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
});

export default SignatureRegistration;
