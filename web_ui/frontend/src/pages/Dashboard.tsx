import { useState, useEffect } from 'react';
import { 
  Gauge, 
  Activity, 
  Clock, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle,
  RefreshCw,
  Network,
  Radio,
  Wifi,
  WifiOff,
  Settings,
  BarChart3,
  Monitor
} from 'lucide-react';
import RQTGraph from '../components/LQTGraph';

interface DashboardData {
  experiments_count: number;
  total_laps: number;
  total_trajectories: number;
  latest_experiment?: {
    name: string;
    last_modified: string;
    laps: number;
  };
}

interface RosStatus {
  ros_available: boolean;
  monitoring_active: boolean;
  active_nodes?: number;
  active_topics?: number;
}

const Dashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [rosStatus, setRosStatus] = useState<RosStatus>({
    ros_available: false,
    monitoring_active: false,
    active_nodes: 0,
    active_topics: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Add timeout to prevent infinite loading
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const [dataResponse, statusResponse] = await Promise.all([
        fetch('/api/data/summary', { signal: controller.signal }),
        fetch('/api/live/status', { signal: controller.signal })
      ]);
      
      clearTimeout(timeoutId);
      
      // Handle data response
      if (dataResponse.ok) {
        const dataResult = await dataResponse.json();
        setData(dataResult);
      } else {
        // If no data available, set empty state instead of keeping loading
        setData({
          experiments_count: 0,
          total_laps: 0,
          total_trajectories: 0
        });
      }
      
      // Handle status response
      if (statusResponse.ok) {
        const statusResult = await statusResponse.json();
        setRosStatus({
          ros_available: statusResult.ros_available || false,
          monitoring_active: statusResult.monitoring_active || false,
          active_nodes: statusResult.active_nodes || 0,
          active_topics: statusResult.active_topics || 0
        });
      } else {
        setRosStatus({
          ros_available: false,
          monitoring_active: false,
          active_nodes: 0,
          active_topics: 0
        });
      }
      
      setError(null);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please check your connection and try again.');
      } else {
        setError('Failed to connect to backend. Make sure the backend server is running.');
      }
      
      // Set default values instead of infinite loading
      setData({
        experiments_count: 0,
        total_laps: 0,
        total_trajectories: 0
      });
      setRosStatus({
        ros_available: false,
        monitoring_active: false,
        active_nodes: 0,
        active_topics: 0
      });
      
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchDashboardData, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setLoading(true);
    fetchDashboardData();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-race-primary" />
        <span className="ml-2 text-gray-600">Loading dashboard...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-red-600">
        <AlertCircle className="h-8 w-8 mr-2" />
        <span>{error}</span>
        <button 
          onClick={handleRefresh}
          className="ml-4 px-4 py-2 bg-race-primary text-white rounded-md hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Race Monitor Dashboard</h1>
        <button
          onClick={handleRefresh}
          className="flex items-center space-x-2 px-4 py-2 bg-race-primary text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-blue-100">
              <Gauge className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Experiments</p>
              <p className="text-2xl font-bold text-gray-900">{data?.experiments_count || 0}</p>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-green-100">
              <Activity className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Laps</p>
              <p className="text-2xl font-bold text-gray-900">{data?.total_laps || 0}</p>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-purple-100">
              <TrendingUp className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Trajectories</p>
              <p className="text-2xl font-bold text-gray-900">{data?.total_trajectories || 0}</p>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-orange-100">
              <Clock className="h-6 w-6 text-orange-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Status</p>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm font-medium text-green-600">Online</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Latest Experiment */}
      {data?.latest_experiment && (
        <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Latest Experiment</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm font-medium text-gray-600">Name</p>
              <p className="text-lg font-semibold text-gray-900">{data.latest_experiment.name}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Laps Completed</p>
              <p className="text-lg font-semibold text-gray-900">{data.latest_experiment.laps}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Last Modified</p>
              <p className="text-lg font-semibold text-gray-900">
                {new Date(data.latest_experiment.last_modified).toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* RQT Graph - ROS2 Node Graph */}
      <RQTGraph realtime={rosStatus.monitoring_active} />

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <button className="flex items-center justify-center space-x-2 p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-race-primary hover:bg-blue-50 transition-colors">
            <Settings className="h-5 w-5 text-gray-600" />
            <span className="text-gray-600">Edit Config</span>
          </button>
          
          <button className="flex items-center justify-center space-x-2 p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-race-primary hover:bg-blue-50 transition-colors">
            <BarChart3 className="h-5 w-5 text-gray-600" />
            <span className="text-gray-600">View Results</span>
          </button>
          
          <button className="flex items-center justify-center space-x-2 p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-race-primary hover:bg-blue-50 transition-colors">
            <Activity className="h-5 w-5 text-gray-600" />
            <span className="text-gray-600">Run Analysis</span>
          </button>
          
          <button className="flex items-center justify-center space-x-2 p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-race-primary hover:bg-blue-50 transition-colors">
            <Monitor className="h-5 w-5 text-gray-600" />
            <span className="text-gray-600">Start Monitor</span>
          </button>
        </div>
      </div>

      {/* System Health */}
      <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">System Health</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-600">Backend API</p>
              <p className="text-lg font-semibold text-green-600">Online</p>
            </div>
            <CheckCircle className="h-6 w-6 text-green-500" />
          </div>
          
          <div className={`flex items-center justify-between p-4 rounded-lg ${
            rosStatus.ros_available ? 'bg-green-50' : 'bg-red-50'
          }`}>
            <div>
              <p className="text-sm font-medium text-gray-600">ROS2 Connection</p>
              <p className={`text-lg font-semibold ${
                rosStatus.ros_available ? 'text-green-600' : 'text-red-600'
              }`}>
                {rosStatus.ros_available ? 'Connected' : 'Disconnected'}
              </p>
            </div>
            {rosStatus.ros_available ? (
              <Wifi className="h-6 w-6 text-green-500" />
            ) : (
              <WifiOff className="h-6 w-6 text-red-500" />
            )}
          </div>
          
          <div className={`flex items-center justify-between p-4 rounded-lg ${
            rosStatus.monitoring_active ? 'bg-blue-50' : 'bg-yellow-50'
          }`}>
            <div>
              <p className="text-sm font-medium text-gray-600">Monitoring</p>
              <p className={`text-lg font-semibold ${
                rosStatus.monitoring_active ? 'text-blue-600' : 'text-yellow-600'
              }`}>
                {rosStatus.monitoring_active ? 'Active' : 'Idle'}
              </p>
            </div>
            {rosStatus.monitoring_active ? (
              <Activity className="h-6 w-6 text-blue-500 animate-pulse" />
            ) : (
              <Clock className="h-6 w-6 text-yellow-500" />
            )}
          </div>
          
          <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-600">EVO Integration</p>
              <p className="text-lg font-semibold text-blue-600">Available</p>
            </div>
            <CheckCircle className="h-6 w-6 text-blue-500" />
          </div>
        </div>
        
        {/* ROS2 Quick Stats */}
        {rosStatus.ros_available && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <h3 className="text-lg font-medium text-gray-900 mb-3">ROS2 System Overview</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <Network className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-600">Active Nodes</p>
                  <p className="text-lg font-semibold text-gray-900">{rosStatus.active_nodes || 'N/A'}</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <Radio className="h-5 w-5 text-green-600" />
                <div>
                  <p className="text-sm text-gray-600">Active Topics</p>
                  <p className="text-lg font-semibold text-gray-900">{rosStatus.active_topics || 'N/A'}</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;