import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput } from 'react-native';

const UserIdSection = ({ userId, onUserIdChange }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>🆔 Tu ID de Usuario:</Text>
      <TextInput
        style={styles.input}
        value={userId}
        onChangeText={onUserIdChange}
        placeholder="Ingresa tu ID (ej: USER001)"
        placeholderTextColor="#999"
        autoCapitalize="characters"
        autoCorrect={false}
        returnKeyType="done"
      />
      {userId ? (
        <Text style={styles.currentUser}>Usuario activo: <Text style={styles.userId}>{userId}</Text></Text>
      ) : (
        <Text style={styles.helper}>Ingresa tu ID para registrar o verificar firmas</Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: 'white',
    marginHorizontal: 20,
    marginVertical: 10,
    borderRadius: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  input: {
    borderWidth: 2,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
    marginBottom: 10,
  },
  currentUser: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  userId: {
    fontWeight: 'bold',
    color: '#2196F3',
  },
  helper: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
    fontStyle: 'italic',
  },
});

export default UserIdSection;
