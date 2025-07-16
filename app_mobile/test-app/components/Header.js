import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Modal, TextInput, Alert } from 'react-native';

const Header = ({ backendUrl, onBackendUrlChange }) => {
  const [modalVisible, setModalVisible] = useState(false);
  const [tempUrl, setTempUrl] = useState(backendUrl);

  const handleSaveUrl = () => {
    if (!tempUrl.trim()) {
      Alert.alert('Error', 'La URL no puede estar vacía');
      return;
    }

    // Validar formato básico de URL
    try {
      new URL(tempUrl);
      onBackendUrlChange(tempUrl.trim());
      setModalVisible(false);
      Alert.alert('✅ Éxito', 'URL del backend actualizada');
    } catch (error) {
      Alert.alert('Error', 'URL inválida. Asegúrate de incluir http:// o https://');
    }
  };

  const handleCancel = () => {
    setTempUrl(backendUrl); // Restaurar valor original
    setModalVisible(false);
  };

  return (
    <>
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Text style={styles.title}>Verificador de Firmas</Text>
          <TouchableOpacity 
            style={styles.configButton}
            onPress={() => setModalVisible(true)}
          >
            <Text style={styles.configIcon}>⚙️</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Modal para cambiar URL */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={handleCancel}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Configurar Backend</Text>
            
            <Text style={styles.inputLabel}>URL del Backend:</Text>
            <TextInput
              style={styles.urlInput}
              value={tempUrl}
              onChangeText={setTempUrl}
              placeholder="http://ejemplo.tunnelmole.net"
              placeholderTextColor="#999"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
            />

            <Text style={styles.helpText}>
              💡 Ejemplo: http://s34dhz-ip-190-239-94-58.tunnelmole.net
            </Text>

            <View style={styles.modalButtons}>
              <TouchableOpacity 
                style={[styles.modalButton, styles.cancelButton]}
                onPress={handleCancel}
              >
                <Text style={styles.cancelButtonText}>Cancelar</Text>
              </TouchableOpacity>

              <TouchableOpacity 
                style={[styles.modalButton, styles.saveButton]}
                onPress={handleSaveUrl}
              >
                <Text style={styles.saveButtonText}>✅ Guardar</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  header: {
    backgroundColor: '#2196F3',
    padding: 20,
    paddingTop: 50,
    alignItems: 'center',
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    position: 'relative',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  configButton: {
    position: 'absolute',
    right: 0,
    padding: 8,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  configIcon: {
    fontSize: 16,
  },
  // Estilos del modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 15,
    padding: 25,
    width: '100%',
    maxWidth: 400,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 10,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  urlInput: {
    borderWidth: 2,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 15,
    fontSize: 14,
    backgroundColor: '#f9f9f9',
    marginBottom: 10,
  },
  helpText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 20,
    lineHeight: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 15,
  },
  modalButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: '#f5f5f5',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  saveButton: {
    backgroundColor: '#4CAF50',
  },
  cancelButtonText: {
    color: '#666',
    fontSize: 16,
    fontWeight: '600',
  },
  saveButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default Header;
