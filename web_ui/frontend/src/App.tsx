import { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Toaster, toast } from 'react-hot-toast';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import NotificationSystem, { Notification } from './components/NotificationSystem';
import Dashboard from './pages/Dashboard';
import Configuration from './pages/Configuration';
import Results from './pages/Results';
import Analysis from './pages/Analysis';
import LiveMonitor from './pages/LiveMonitor';
import RosNodes from './pages/RosNodes';
import RosTopics from './pages/RosTopics';

interface RosStatus {
  ros_available: boolean;
  monitoring_active: boolean;
  timestamp: string;
  active_nodes?: number;
  active_topics?: number;
}

interface NotificationCounts {
  unread_results: number;
  unread_analysis: number;
}

function App() {
  const [rosStatus, setRosStatus] = useState<RosStatus>({
    ros_available: false,
    monitoring_active: false,
    timestamp: new Date().toISOString(),
    active_nodes: 0,
    active_topics: 0
  });

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [notificationCounts, setNotificationCounts] = useState<NotificationCounts>({
    unread_results: 0,
    unread_analysis: 0
  });

  // Mock data for demonstration
  useEffect(() => {
    // Simulate some notifications and counts
    setNotificationCounts({
      unread_results: 2,
      unread_analysis: 1
    });

    // Add sample notifications
    setNotifications([
      {
        id: '1',
        type: 'success',
        title: 'Race Completed',
        message: 'Race session finished successfully. Results are now available.',
        timestamp: new Date(),
        read: false,
        actions: [
          {
            label: 'View Results',
            action: () => window.location.href = '/results',
            primary: true
          }
        ]
      },
      {
        id: '2',
        type: 'info',
        title: 'EVO Analysis Ready',
        message: 'New trajectory analysis has been completed.',
        timestamp: new Date(Date.now() - 300000), // 5 minutes ago
        read: false,
        actions: [
          {
            label: 'View Analysis',
            action: () => window.location.href = '/analysis',
            primary: true
          }
        ]
      }
    ]);
  }, []);

  // Check ROS2 status periodically
  useEffect(() => {
    const checkRosStatus = async () => {
      try {
        const response = await fetch('/api/live/status');
        if (response.ok) {
          const data = await response.json();
          setRosStatus({
            ros_available: data.ros_available || false,
            monitoring_active: data.monitoring_active || false,
            timestamp: new Date().toISOString(),
            active_nodes: data.active_nodes || 5, // Mock data
            active_topics: data.active_topics || 12 // Mock data
          });
        }
      } catch (error) {
        console.error('Failed to fetch ROS status:', error);
        setRosStatus(prev => ({
          ...prev,
          ros_available: false,
          monitoring_active: false,
          timestamp: new Date().toISOString(),
          active_nodes: 5, // Mock data
          active_topics: 12 // Mock data
        }));
      }
    };

    // Check immediately
    checkRosStatus();

    // Check every 5 seconds
    const interval = setInterval(checkRosStatus, 5000);

    return () => clearInterval(interval);
  }, []);

  // Handle race completion notification
  const handleRaceComplete = () => {
    const newNotification: Notification = {
      id: Date.now().toString(),
      type: 'success',
      title: 'Race Completed!',
      message: 'The race has finished successfully. New results and analysis are available.',
      timestamp: new Date(),
      read: false,
      actions: [
        {
          label: 'View Results',
          action: () => window.location.href = '/results',
          primary: true
        },
        {
          label: 'View Analysis',
          action: () => window.location.href = '/analysis'
        }
      ]
    };

    setNotifications(prev => [newNotification, ...prev]);
    setNotificationCounts(prev => ({
      unread_results: prev.unread_results + 1,
      unread_analysis: prev.unread_analysis + 1
    }));

    toast.success('Race completed! Check your results.', {
      duration: 5000,
      icon: '🏁'
    });
  };

  const handleMarkNotificationAsRead = (id: string) => {
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
  };

  const handleRemoveNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-50 flex">
      <Sidebar 
        collapsed={sidebarCollapsed} 
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        rosStatus={rosStatus}
        notificationCounts={notificationCounts}
      />
      
      <div className={`flex-1 flex flex-col transition-all duration-300 ${
        sidebarCollapsed ? 'ml-16' : 'ml-64'
      }`}>
        <div className="flex items-center justify-between bg-white shadow-sm border-b border-gray-200 px-6 py-4">
          <Header rosStatus={rosStatus} onRaceComplete={handleRaceComplete} />
          
          <NotificationSystem
            notifications={notifications}
            onMarkAsRead={handleMarkNotificationAsRead}
            onRemove={handleRemoveNotification}
          />
        </div>
        
        <main className="flex-1 p-6 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/config" element={<Configuration />} />
            <Route path="/results" element={<Results />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/live" element={<LiveMonitor />} />
            <Route path="/ros-nodes" element={<RosNodes />} />
            <Route path="/ros-topics" element={<RosTopics />} />
          </Routes>
        </main>
      </div>

      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
    </div>
  );
}

export default App;