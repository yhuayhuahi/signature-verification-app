import React from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity 
} from 'react-native';

const TabNavigation = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'register', title: 'Registrar', icon: '📝' },
    { id: 'verify', title: 'Verificar', icon: '🔍' },
    { id: 'stats', title: 'Estadísticas', icon: '📊' },
  ];

  return (
    <View style={styles.container}>
      {tabs.map((tab) => (
        <TouchableOpacity
          key={tab.id}
          style={[
            styles.tab,
            activeTab === tab.id && styles.activeTab
          ]}
          onPress={() => onTabChange(tab.id)}
        >
          <Text style={[
            styles.tabIcon,
            activeTab === tab.id && styles.activeTabIcon
          ]}>
            {tab.icon}
          </Text>
          <Text style={[
            styles.tabText,
            activeTab === tab.id && styles.activeTabText
          ]}>
            {tab.title}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: 'white',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  tab: {
    flex: 1,
    paddingVertical: 15,
    alignItems: 'center',
    backgroundColor: 'white',
  },
  activeTab: {
    backgroundColor: '#E3F2FD',
    borderTopWidth: 3,
    borderTopColor: '#2196F3',
  },
  tabIcon: {
    fontSize: 20,
    marginBottom: 5,
  },
  activeTabIcon: {
    transform: [{ scale: 1.1 }],
  },
  tabText: {
    fontSize: 12,
    color: '#666',
    fontWeight: '600',
  },
  activeTabText: {
    color: '#2196F3',
    fontWeight: 'bold',
  },
});

export default TabNavigation;
