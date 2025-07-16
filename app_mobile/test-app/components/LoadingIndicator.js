import React from 'react';
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';

const LoadingIndicator = ({ message = 'Procesando...' }) => {
  return (
    <View style={styles.loadingContainer}>
      <ActivityIndicator size="large" color="#2196F3" />
      <Text style={styles.loadingText}>{message}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  loadingContainer: {
    margin: 20,
    padding: 25,
    backgroundColor: 'white',
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#333',
    fontWeight: '600',
    textAlign: 'center',
  },
});

export default LoadingIndicator;
