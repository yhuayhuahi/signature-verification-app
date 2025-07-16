import { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Alert } from 'react-native';

export const useImageHandler = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const pickImageFromGallery = async () => {
    try {
      setIsLoading(true);
      
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        const image = result.assets[0];
        setSelectedImage(image);
        console.log('✅ Imagen seleccionada desde galería:', image.uri);
        return image;
      }
    } catch (error) {
      console.error('❌ Error seleccionando imagen:', error);
      Alert.alert('Error', 'No se pudo seleccionar la imagen');
    } finally {
      setIsLoading(false);
    }
    return null;
  };

  const takePhotoWithCamera = async () => {
    try {
      setIsLoading(true);
      
      // Verificar permisos de cámara
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permisos', 'Se necesita permiso para usar la cámara');
        return null;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        const image = result.assets[0];
        setSelectedImage(image);
        console.log('✅ Foto tomada con cámara:', image.uri);
        return image;
      }
    } catch (error) {
      console.error('❌ Error tomando foto:', error);
      Alert.alert('Error', 'No se pudo tomar la foto');
    } finally {
      setIsLoading(false);
    }
    return null;
  };

  const clearImage = () => {
    setSelectedImage(null);
  };

  return {
    selectedImage,
    isLoading,
    pickImageFromGallery,
    takePhotoWithCamera,
    clearImage,
  };
};
