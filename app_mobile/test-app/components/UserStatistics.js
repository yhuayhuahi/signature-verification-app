import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator } from 'react-native';

const Statistics = ({ userId, stats, onRefreshStats, isLoading }) => {
  if (!userId?.trim()) {
    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>📊 Mis Estadísticas</Text>
        </View>
        <View style={styles.placeholderContainer}>
          <Text style={styles.placeholderText}>
            🆔 Ingresa tu ID para ver tus estadísticas
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>📊 Mis Estadísticas</Text>
        <TouchableOpacity 
          style={[styles.refreshButton, isLoading && styles.buttonDisabled]}
          onPress={() => onRefreshStats(userId)}
          disabled={isLoading}
        >
          {isLoading ? (
            <ActivityIndicator color="white" size="small" />
          ) : (
            <Text style={styles.refreshText}>🔄</Text>
          )}
        </TouchableOpacity>
      </View>

      {stats ? (
        <View style={styles.statsContent}>
          <View style={styles.userInfo}>
            <Text style={styles.userLabel}>Usuario:</Text>
            <Text style={styles.userValue}>{stats.user_id || userId}</Text>
          </View>

          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{stats.registered_signatures || 0}</Text>
              <Text style={styles.statLabel}>Firmas{'\n'}Registradas</Text>
            </View>
          </View>

          {stats.message && (
            <View style={styles.messageContainer}>
              <Text style={styles.messageText}>{stats.message}</Text>
            </View>
          )}
        </View>
      ) : (
        <View style={styles.loadingContainer}>
          <ActivityIndicator color="#2196F3" size="large" />
          <Text style={styles.loadingText}>Cargando estadísticas...</Text>
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  refreshButton: {
    backgroundColor: '#2196F3',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 8,
    elevation: 2,
  },
  refreshText: {
    color: 'white',
    fontSize: 16,
  },
  buttonDisabled: {
    backgroundColor: '#BDBDBD',
  },
  placeholderContainer: {
    padding: 30,
    alignItems: 'center',
  },
  placeholderText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
  statsContent: {
    gap: 15,
  },
  userInfo: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#E3F2FD',
    padding: 15,
    borderRadius: 8,
    gap: 10,
  },
  userLabel: {
    fontSize: 16,
    color: '#666',
  },
  userValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  statsGrid: {
    alignItems: 'center',
  },
  statCard: {
    backgroundColor: '#F3E5F5',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    minWidth: 120,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  statNumber: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#9C27B0',
    marginBottom: 5,
  },
  statLabel: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 18,
  },
  messageContainer: {
    backgroundColor: '#E8F5E8',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  messageText: {
    fontSize: 14,
    color: '#2E7D32',
    fontStyle: 'italic',
  },
  loadingContainer: {
    padding: 30,
    alignItems: 'center',
    gap: 10,
  },
  loadingText: {
    fontSize: 16,
    color: '#666',
  },
});

export default Statistics;
