import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator } from 'react-native';

const ImagePickerButtons = ({ onPickImage, onTakePhoto, isUploading }) => {
  return (
    <View style={styles.buttonContainer}>
      <View style={styles.buttonRow}>
        <TouchableOpacity 
          style={[styles.button, styles.galleryButton, isUploading && styles.buttonDisabled]} 
          onPress={onPickImage}
          disabled={isUploading}
        >
          <Text style={styles.buttonIcon}>📁</Text>
          <Text style={styles.buttonText}>Seleccionar{'\n'}de Galería</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.button, styles.cameraButton, isUploading && styles.buttonDisabled]} 
          onPress={onTakePhoto}
          disabled={isUploading}
        >
          {isUploading ? (
            <View style={styles.buttonContent}>
              <ActivityIndicator color="white" size="small" />
              <Text style={styles.buttonText}>Procesando...</Text>
            </View>
          ) : (
            <>
              <Text style={styles.buttonIcon}>📷</Text>
              <Text style={styles.buttonText}>Tomar{'\n'}Foto</Text>
            </>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  buttonContainer: {
    padding: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 15,
  },
  button: {
    flex: 1,
    aspectRatio: 1, // Hace los botones cuadrados
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    paddingVertical: 15,
    paddingHorizontal: 10,
  },
  galleryButton: {
    backgroundColor: '#4CAF50',
  },
  cameraButton: {
    backgroundColor: '#FF9800',
  },
  buttonDisabled: {
    backgroundColor: '#BDBDBD',
  },
  buttonContent: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    lineHeight: 18,
  },
});

export default ImagePickerButtons;
