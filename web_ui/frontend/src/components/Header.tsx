import { useLocation } from 'react-router-dom';
import { 
  Power,
  PowerOff,
  Wifi,
  WifiOff,
  Activity
} from 'lucide-react';

interface RosStatus {
  ros_available: boolean;
  monitoring_active: boolean;
  timestamp: string;
}

interface HeaderProps {
  rosStatus: RosStatus;
  onRaceComplete?: () => void;
}

const Header: React.FC<HeaderProps> = ({ rosStatus }) => {
  const location = useLocation();

  const getPageTitle = () => {
    switch (location.pathname) {
      case '/': return 'Dashboard';
      case '/live': return 'Live Monitor';
      case '/results': return 'Race Results';
      case '/analysis': return 'EVO Analysis & Comparison';
      case '/ros-nodes': return 'ROS2 Nodes';
      case '/ros-topics': return 'ROS2 Topics';
      case '/config': return 'Configuration';
      default: return 'Race Monitor';
    }
  };

  const startMonitoring = async () => {
    try {
      await fetch('/api/live/start', { method: 'POST' });
    } catch (error) {
      console.error('Failed to start monitoring:', error);
    }
  };

  const stopMonitoring = async () => {
    try {
      await fetch('/api/live/stop', { method: 'POST' });
    } catch (error) {
      console.error('Failed to stop monitoring:', error);
    }
  };

  return (
    <div className="flex items-center justify-between flex-1">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">{getPageTitle()}</h1>
        <p className="text-sm text-gray-500 mt-1">
          Real-time ROS2 race monitoring and analysis
        </p>
      </div>

      {/* Status and Controls */}
      <div className="flex items-center space-x-6">
        {/* ROS2 Status */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            {rosStatus.ros_available ? (
              <Wifi className="h-5 w-5 text-green-500" />
            ) : (
              <WifiOff className="h-5 w-5 text-red-500" />
            )}
            <span className={`text-sm font-medium ${
              rosStatus.ros_available ? 'text-green-600' : 'text-red-600'
            }`}>
              {rosStatus.ros_available ? 'ROS2 Connected' : 'ROS2 Disconnected'}
            </span>
          </div>

          {/* Monitoring Control */}
          {rosStatus.ros_available && (
            <div className="flex items-center space-x-2">
              <Activity className={`h-4 w-4 ${
                rosStatus.monitoring_active ? 'text-green-500 animate-pulse' : 'text-gray-400'
              }`} />
              <span className={`text-sm ${
                rosStatus.monitoring_active ? 'text-green-600' : 'text-gray-500'
              }`}>
                {rosStatus.monitoring_active ? 'Monitoring' : 'Idle'}
              </span>
              
              {rosStatus.monitoring_active ? (
                <button
                  onClick={stopMonitoring}
                  className="ml-2 px-3 py-1 bg-red-100 text-red-700 rounded-md text-xs font-medium hover:bg-red-200 transition-colors flex items-center space-x-1"
                >
                  <PowerOff className="h-3 w-3" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  onClick={startMonitoring}
                  className="ml-2 px-3 py-1 bg-green-100 text-green-700 rounded-md text-xs font-medium hover:bg-green-200 transition-colors flex items-center space-x-1"
                >
                  <Power className="h-3 w-3" />
                  <span>Start</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Header;