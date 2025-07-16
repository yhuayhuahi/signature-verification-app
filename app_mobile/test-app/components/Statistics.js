import React from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  ScrollView,
  TouchableOpacity 
} from 'react-native';

const Statistics = ({ 
  stats, 
  onRefreshStats, 
  isLoading 
}) => {
  if (!stats) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>📊 Estadísticas</Text>
        <TouchableOpacity 
          style={styles.refreshButton}
          onPress={onRefreshStats}
          disabled={isLoading}
        >
          <Text style={styles.refreshButtonText}>
            {isLoading ? '🔄 Cargando...' : '🔄 Cargar Estadísticas'}
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>📊 Estadísticas de Firmas</Text>
        <TouchableOpacity 
          style={styles.refreshButton}
          onPress={onRefreshStats}
          disabled={isLoading}
        >
          <Text style={styles.refreshButtonText}>
            {isLoading ? '🔄 Actualizando...' : '🔄 Actualizar'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Estadísticas Generales */}
      <View style={styles.statsCard}>
        <Text style={styles.cardTitle}>📈 Resumen General</Text>
        
        <View style={styles.statRow}>
          <Text style={styles.statLabel}>👥 Total de Usuarios:</Text>
          <Text style={styles.statValue}>{stats.totalUsers || 0}</Text>
        </View>
        
        <View style={styles.statRow}>
          <Text style={styles.statLabel}>📝 Total de Firmas:</Text>
          <Text style={styles.statValue}>{stats.totalSignatures || 0}</Text>
        </View>
        
        <View style={styles.statRow}>
          <Text style={styles.statLabel}>🔍 Verificaciones:</Text>
          <Text style={styles.statValue}>{stats.totalVerifications || 0}</Text>
        </View>
      </View>

      {/* Lista de Usuarios Registrados */}
      {stats.registeredUsers && stats.registeredUsers.length > 0 && (
        <View style={styles.statsCard}>
          <Text style={styles.cardTitle}>👤 Usuarios Registrados</Text>
          {stats.registeredUsers.map((user, index) => (
            <View key={index} style={styles.userRow}>
              <View style={styles.userInfo}>
                <Text style={styles.userId}>🆔 {user.id}</Text>
                <Text style={styles.userDate}>
                  📅 {new Date(user.registeredAt).toLocaleDateString()}
                </Text>
              </View>
              <Text style={styles.signatureCount}>
                {user.signatureCount || 1} firma{user.signatureCount > 1 ? 's' : ''}
              </Text>
            </View>
          ))}
        </View>
      )}

      {/* Estadísticas de Verificaciones Recientes */}
      {stats.recentVerifications && stats.recentVerifications.length > 0 && (
        <View style={styles.statsCard}>
          <Text style={styles.cardTitle}>🕒 Verificaciones Recientes</Text>
          {stats.recentVerifications.slice(0, 10).map((verification, index) => (
            <View key={index} style={styles.verificationRow}>
              <View style={styles.verificationInfo}>
                <Text style={styles.verificationUser}>
                  🆔 {verification.userId}
                </Text>
                <Text style={styles.verificationDate}>
                  📅 {new Date(verification.timestamp).toLocaleString()}
                </Text>
              </View>
              <View style={styles.verificationResult}>
                <Text style={[
                  styles.verificationStatus,
                  verification.isMatch ? styles.successText : styles.errorText
                ]}>
                  {verification.isMatch ? '✅' : '❌'}
                </Text>
                <Text style={styles.similarityText}>
                  {(verification.similarity * 100).toFixed(1)}%
                </Text>
              </View>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    margin: 20,
    marginBottom: 10,
    alignItems: 'center',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  refreshButton: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  refreshButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  statsCard: {
    backgroundColor: 'white',
    margin: 20,
    marginTop: 10,
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  statLabel: {
    fontSize: 16,
    color: '#666',
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  userRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  userInfo: {
    flex: 1,
  },
  userId: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  userDate: {
    fontSize: 12,
    color: '#999',
  },
  signatureCount: {
    fontSize: 14,
    color: '#2196F3',
    fontWeight: 'bold',
  },
  verificationRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  verificationInfo: {
    flex: 1,
  },
  verificationUser: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  verificationDate: {
    fontSize: 12,
    color: '#999',
  },
  verificationResult: {
    alignItems: 'center',
  },
  verificationStatus: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  successText: {
    color: '#4CAF50',
  },
  errorText: {
    color: '#F44336',
  },
  similarityText: {
    fontSize: 12,
    color: '#666',
  },
});

export default Statistics;
