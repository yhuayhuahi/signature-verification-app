import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity } from 'react-native';

const UserIdInput = ({ onSubmit, buttonText = "Continuar", buttonColor = "#2196F3", placeholder = "Ingresa tu ID de usuario" }) => {
  const [userId, setUserId] = useState('');

  const handleSubmit = () => {
    if (userId.trim()) {
      onSubmit(userId.trim());
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.label}>ID de Usuario:</Text>
      <TextInput
        style={styles.input}
        value={userId}
        onChangeText={setUserId}
        placeholder={placeholder}
        placeholderTextColor="#999"
        autoCapitalize="characters"
        autoCorrect={false}
        returnKeyType="done"
        onSubmitEditing={handleSubmit}
      />
      <TouchableOpacity 
        style={[styles.button, { backgroundColor: buttonColor }, !userId.trim() && styles.buttonDisabled]}
        onPress={handleSubmit}
        disabled={!userId.trim()}
      >
        <Text style={styles.buttonText}>{buttonText}</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    gap: 15,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  input: {
    borderWidth: 2,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    backgroundColor: 'white',
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  button: {
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  buttonDisabled: {
    backgroundColor: '#BDBDBD',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default UserIdInput;
