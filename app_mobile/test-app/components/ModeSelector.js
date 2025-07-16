import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';

const ModeSelector = ({ selectedMode, onModeChange, userId }) => {
  if (!userId?.trim()) {
    return (
      <View style={styles.container}>
        <Text style={styles.disabledText}>
          🆔 Ingresa tu ID para seleccionar una acción
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>¿Qué deseas hacer?</Text>
      
      <View style={styles.optionsContainer}>
        <TouchableOpacity
          style={[
            styles.option,
            styles.registerOption,
            selectedMode === 'register' && styles.selectedOption
          ]}
          onPress={() => onModeChange('register')}
        >
          <Text style={styles.optionIcon}>✍️</Text>
          <Text style={[
            styles.optionText,
            selectedMode === 'register' && styles.selectedText
          ]}>
            Registrar Firma
          </Text>
          <Text style={[
            styles.optionDescription,
            selectedMode === 'register' && styles.selectedDescription
          ]}>
            Guardar una nueva firma de referencia
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.option,
            styles.verifyOption,
            selectedMode === 'verify' && styles.selectedOption
          ]}
          onPress={() => onModeChange('verify')}
        >
          <Text style={styles.optionIcon}>🔍</Text>
          <Text style={[
            styles.optionText,
            selectedMode === 'verify' && styles.selectedText
          ]}>
            Verificar Firma
          </Text>
          <Text style={[
            styles.optionDescription,
            selectedMode === 'verify' && styles.selectedDescription
          ]}>
            Comparar con firmas registradas
          </Text>
        </TouchableOpacity>
      </View>

      {selectedMode && (
        <View style={styles.selectedInfo}>
          <Text style={styles.selectedInfoText}>
            Modo seleccionado: <Text style={styles.selectedModeText}>
              {selectedMode === 'register' ? 'Registrar Firma' : 'Verificar Firma'}
            </Text>
          </Text>
        </View>
      )}
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
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 20,
  },
  disabledText: {
    fontSize: 16,
    color: '#999',
    textAlign: 'center',
    fontStyle: 'italic',
    padding: 20,
  },
  optionsContainer: {
    gap: 15,
  },
  option: {
    borderWidth: 2,
    borderColor: '#E0E0E0',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    backgroundColor: '#F9F9F9',
  },
  registerOption: {
    borderColor: '#4CAF50',
  },
  verifyOption: {
    borderColor: '#FF9800',
  },
  selectedOption: {
    backgroundColor: '#E8F5E8',
    borderColor: '#4CAF50',
    borderWidth: 3,
  },
  optionIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  optionText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  selectedText: {
    color: '#2E7D32',
  },
  optionDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  selectedDescription: {
    color: '#4CAF50',
    fontWeight: '500',
  },
  selectedInfo: {
    marginTop: 20,
    padding: 15,
    backgroundColor: '#E3F2FD',
    borderRadius: 8,
    alignItems: 'center',
  },
  selectedInfoText: {
    fontSize: 14,
    color: '#1976D2',
  },
  selectedModeText: {
    fontWeight: 'bold',
    color: '#0D47A1',
  },
});

export default ModeSelector;
