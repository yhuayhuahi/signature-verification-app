import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator } from 'react-native';

const ActionButtons = ({ 
  selectedImage, 
  userId,
  onRegisterSignature, 
  onVerifySignature, 
  isLoading 
}) => {
  if (!selectedImage) {
    return (
      <View style={styles.placeholderContainer}>
        <Text style={styles.placeholderText}>
          📸 Selecciona una imagen para continuar
        </Text>
      </View>
    );
  }

  if (!userId?.trim()) {
    return (
      <View style={styles.placeholderContainer}>
        <Text style={styles.placeholderText}>
          🆔 Ingresa tu ID de usuario para continuar
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>¿Qué deseas hacer?</Text>
      
      {/* Botón de Registrar */}
      <View style={styles.actionSection}>
        <View style={styles.actionInfo}>
          <Text style={styles.actionTitle}>✍️ Registrar Nueva Firma</Text>
          <Text style={styles.actionDescription}>
            Guarda esta imagen como una firma de referencia para tu ID
          </Text>
        </View>
        <TouchableOpacity 
          style={[styles.actionButton, styles.registerButton, isLoading && styles.buttonDisabled]}
          onPress={() => onRegisterSignature(selectedImage, userId)}
          disabled={isLoading}
        >
          {isLoading ? (
            <View style={styles.buttonContent}>
              <ActivityIndicator color="white" size="small" />
              <Text style={styles.buttonText}>Registrando...</Text>
            </View>
          ) : (
            <Text style={styles.buttonText}>💾 Registrar</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Separador */}
      <View style={styles.separator} />

      {/* Botón de Verificar */}
      <View style={styles.actionSection}>
        <View style={styles.actionInfo}>
          <Text style={styles.actionTitle}>🔍 Verificar Firma</Text>
          <Text style={styles.actionDescription}>
            Compara esta imagen con las firmas registradas de tu ID
          </Text>
        </View>
        <TouchableOpacity 
          style={[styles.actionButton, styles.verifyButton, isLoading && styles.buttonDisabled]}
          onPress={() => onVerifySignature(selectedImage, userId)}
          disabled={isLoading}
        >
          {isLoading ? (
            <View style={styles.buttonContent}>
              <ActivityIndicator color="white" size="small" />
              <Text style={styles.buttonText}>Verificando...</Text>
            </View>
          ) : (
            <Text style={styles.buttonText}>� Verificar</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
  },
  placeholderContainer: {
    padding: 20,
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 20,
  },
  actionSection: {
    marginVertical: 10,
  },
  actionInfo: {
    marginBottom: 15,
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  actionDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 18,
  },
  actionButton: {
    padding: 18,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    minHeight: 55,
  },
  registerButton: {
    backgroundColor: '#4CAF50',
  },
  verifyButton: {
    backgroundColor: '#FF9800',
  },
  buttonDisabled: {
    backgroundColor: '#BDBDBD',
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  separator: {
    height: 1,
    backgroundColor: '#E0E0E0',
    marginVertical: 20,
  },
});

export default ActionButtons;
