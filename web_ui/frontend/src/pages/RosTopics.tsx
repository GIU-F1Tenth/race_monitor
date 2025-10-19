import { useState, useEffect } from 'react';
import { 
  Radio, 
  Activity, 
  RefreshCw, 
  Eye,
  EyeOff,
  Clock,
  Database,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';

interface RosTopic {
  name: string;
  type: string;
  publishers: string[];
  subscribers: string[];
  frequency: number;
  last_message?: string;
  message_count: number;
  status: 'active' | 'inactive' | 'slow';
}

const RosTopics: React.FC = () => {
  const [topics, setTopics] = useState<RosTopic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [monitoringTopics, setMonitoringTopics] = useState<Set<string>>(new Set());

  const fetchTopics = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Add timeout to prevent infinite loading
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch('/api/ros/topics', { signal: controller.signal });
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const data = await response.json();
        setTopics(data.topics || []);
        setLastUpdate(new Date());
      } else {
        // If API not available, show empty state instead of mock data
        setTopics([]);
        setError('ROS2 topics API not available. Make sure ROS2 is running and the backend is connected.');
      }
    } catch (err: any) {
      setTopics([]);
      if (err.name === 'AbortError') {
        setError('Request timed out. Please check your connection and try again.');
      } else {
        setError('Failed to connect to ROS2 system. Check if ROS2 is running.');
      }
      console.error('Error fetching topics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTopics();
    
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchTopics, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600';
      case 'slow': return 'text-yellow-600';
      case 'inactive': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'slow': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'inactive': return <XCircle className="h-4 w-4 text-red-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const toggleTopicMonitoring = (topicName: string) => {
    const newMonitoring = new Set(monitoringTopics);
    if (newMonitoring.has(topicName)) {
      newMonitoring.delete(topicName);
    } else {
      newMonitoring.add(topicName);
    }
    setMonitoringTopics(newMonitoring);
  };

  const formatFrequency = (freq: number) => {
    if (freq === 0) return 'No data';
    if (freq < 1) return `${(freq * 1000).toFixed(0)}ms`;
    return `${freq.toFixed(1)} Hz`;
  };

  if (loading && topics.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
        <span className="ml-2 text-gray-600">Loading ROS2 topics...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">ROS2 Topics</h1>
          <p className="text-gray-600">Monitor active topics and their message flow</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-500">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
          <button
            onClick={fetchTopics}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-yellow-600 mr-2" />
            <span className="text-yellow-800">{error}</span>
          </div>
        </div>
      )}

      {topics.length === 0 && !loading && !error && (
        <div className="text-center py-12">
          <Radio className="h-12 w-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No ROS2 Topics Found</h3>
          <p className="text-gray-500">
            No active topics detected. Make sure ROS2 nodes are running and publishing data.
          </p>
        </div>
      )}

      {topics.length > 0 && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Topics</p>
                  <p className="text-2xl font-bold text-gray-900">{topics.length}</p>
                </div>
                <Radio className="h-8 w-8 text-blue-600" />
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Active Topics</p>
                  <p className="text-2xl font-bold text-green-600">
                    {topics.filter(t => t.status === 'active').length}
                  </p>
                </div>
                <Activity className="h-8 w-8 text-green-600" />
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Slow Topics</p>
                  <p className="text-2xl font-bold text-yellow-600">
                    {topics.filter(t => t.status === 'slow').length}
                  </p>
                </div>
                <Clock className="h-8 w-8 text-yellow-600" />
              </div>
            </div>
            
            <div className="bg-white p-4 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Messages</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {topics.reduce((sum, t) => sum + t.message_count, 0).toLocaleString()}
                  </p>
                </div>
                <Database className="h-8 w-8 text-blue-600" />
              </div>
            </div>
          </div>

          {/* Topics Table */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Topic Details</h3>
            </div>
            
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Topic Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Frequency
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Publishers
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Subscribers
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Messages
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Monitor
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {topics.map((topic) => (
                    <tr key={topic.name} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="font-medium text-gray-900">{topic.name}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-600 font-mono">{topic.type}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(topic.status)}
                          <span className={`text-sm font-medium ${getStatusColor(topic.status)}`}>
                            {topic.status}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{formatFrequency(topic.frequency)}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-600">
                          {topic.publishers.length > 0 ? topic.publishers.join(', ') : 'None'}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-600">
                          {topic.subscribers.length > 0 ? topic.subscribers.join(', ') : 'None'}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          {topic.message_count.toLocaleString()}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button
                          onClick={() => toggleTopicMonitoring(topic.name)}
                          className={`p-2 rounded-lg transition-colors ${
                            monitoringTopics.has(topic.name)
                              ? 'bg-green-100 text-green-600 hover:bg-green-200'
                              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                          }`}
                          title={monitoringTopics.has(topic.name) ? 'Stop monitoring' : 'Start monitoring'}
                        >
                          {monitoringTopics.has(topic.name) ? (
                            <Eye className="h-4 w-4" />
                          ) : (
                            <EyeOff className="h-4 w-4" />
                          )}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default RosTopics;