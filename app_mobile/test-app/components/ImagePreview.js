import React from 'react';
import { StyleSheet, Text, View, Image } from 'react-native';

const ImagePreview = ({ image }) => {
  if (!image) return null;

  return (
    <View style={styles.imageContainer}>
      <Text style={styles.sectionTitle}>✅ Imagen Seleccionada:</Text>
      <Image source={{ uri: image.uri }} style={styles.image} />
      <View style={styles.imageInfoContainer}>
        <Text style={styles.imageInfo}>
          📊 Tamaño: {image.width}x{image.height}
        </Text>
        <Text style={styles.imageInfo}>
          📁 Archivo: {image.fileName || 'Sin nombre'}
        </Text>
        <Text style={styles.imageInfo}>
          📱 Tipo: {image.mimeType || 'image/jpeg'}
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  imageContainer: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  image: {
    width: 250,
    height: 200,
    borderRadius: 8,
    marginBottom: 15,
    resizeMode: 'cover',
  },
  imageInfoContainer: {
    alignItems: 'center',
    gap: 5,
  },
  imageInfo: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
});

export default ImagePreview;
