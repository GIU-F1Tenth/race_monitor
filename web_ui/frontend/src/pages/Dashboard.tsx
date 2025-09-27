import React, { useState, useEffect } from 'react';
import { 
  Gauge, 
  Activity, 
  Clock, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle,
  RefreshCw
} from 'lucide-react';
import axios from 'axios';

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

const Dashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      const response = await axios.get('/api/data/summary');
      setData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch dashboard data');
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-600">Backend API</p>
              <p className="text-lg font-semibold text-green-600">Online</p>
            </div>
            <CheckCircle className="h-6 w-6 text-green-500" />
          </div>
          
          <div className="flex items-center justify-between p-4 bg-yellow-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-600">ROS2 Connection</p>
              <p className="text-lg font-semibold text-yellow-600">Checking...</p>
            </div>
            <RefreshCw className="h-6 w-6 text-yellow-500 animate-spin" />
          </div>
          
          <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-600">EVO Integration</p>
              <p className="text-lg font-semibold text-blue-600">Available</p>
            </div>
            <CheckCircle className="h-6 w-6 text-blue-500" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;