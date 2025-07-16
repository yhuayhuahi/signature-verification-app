import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  Alert 
} from 'react-native';
import { ImagePreview, ActionButton, LoadingIndicator } from './';

const SignatureVerification = ({ 
  selectedImage, 
  onVerifySignature, 
  isLoading,
  verificationResult 
}) => {
  const [targetUserId, setTargetUserId] = useState('');
  const [isVerifying, setIsVerifying] = useState(false);

  const handleVerify = async () => {
    if (!selectedImage) {
      Alert.alert('Error', 'Primero selecciona una imagen para verificar');
      return;
    }

    if (!targetUserId.trim()) {
      Alert.alert('Error', 'Ingresa el ID del usuario a verificar');
      return;
    }

    setIsVerifying(true);
    try {
      await onVerifySignature(selectedImage, targetUserId.trim());
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🔍 Verificar Firma</Text>
      
      <ImagePreview image={selectedImage} />
      
      {selectedImage && (
        <View style={styles.formContainer}>
          <Text style={styles.label}>🎯 ID de Usuario a Verificar:</Text>
          <TextInput
            style={styles.textInput}
            value={targetUserId}
            onChangeText={setTargetUserId}
            placeholder="Ingresa ID del usuario"
            maxLength={20}
            autoCapitalize="characters"
          />
          
          {isVerifying && (
            <LoadingIndicator message="Verificando firma..." />
          )}
          
          <ActionButton
            title="🔎 Verificar Firma"
            onPress={handleVerify}
            isLoading={isVerifying}
            disabled={isVerifying || isLoading}
          />
        </View>
      )}
      
      {/* Resultado de Verificación */}
      {verificationResult && (
        <View style={styles.resultContainer}>
          <Text style={[
            styles.resultTitle,
            verificationResult.isMatch ? styles.successText : styles.errorText
          ]}>
            {verificationResult.isMatch ? '✅ Firma Válida' : '❌ Firma No Válida'}
          </Text>
          
          <View style={styles.resultDetails}>
            <Text style={styles.resultLabel}>📊 Similitud:</Text>
            <Text style={[
              styles.similarityValue,
              verificationResult.isMatch ? styles.highSimilarity : styles.lowSimilarity
            ]}>
              {(verificationResult.similarity * 100).toFixed(1)}%
            </Text>
          </View>
          
          <Text style={styles.resultInfo}>
            🆔 Usuario: {verificationResult.userId}
          </Text>
          
          <Text style={styles.resultInfo}>
            📅 Verificado: {new Date().toLocaleString()}
          </Text>
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
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  textInput: {
    borderWidth: 2,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
    marginBottom: 20,
  },
  resultContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 15,
  },
  successText: {
    color: '#4CAF50',
  },
  errorText: {
    color: '#F44336',
  },
  resultDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
    backgroundColor: '#f5f5f5',
    padding: 15,
    borderRadius: 8,
  },
  resultLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  similarityValue: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  highSimilarity: {
    color: '#4CAF50',
  },
  lowSimilarity: {
    color: '#F44336',
  },
  resultInfo: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
    textAlign: 'center',
  },
});

export default SignatureVerification;
