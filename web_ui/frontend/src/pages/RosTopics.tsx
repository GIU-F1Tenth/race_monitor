import { useState, useEffect } from 'react';
import { 
  Radio, 
  Activity, 
  RefreshCw, 
  Eye,
  EyeOff,
  Clock,
  TrendingUp,
  Database
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
      // Mock ROS2 topics data - in real implementation, this would call `/api/ros/topics`
      const mockTopics: RosTopic[] = [
        {
          name: '/odom',
          type: 'nav_msgs/msg/Odometry',
          publishers: ['car_controller'],
          subscribers: ['race_monitor', 'trajectory_analyzer'],
          frequency: 100.0,
          last_message: new Date().toISOString(),
          message_count: 15432,
          status: 'active'
        },
        {
          name: '/scan',
          type: 'sensor_msgs/msg/LaserScan',
          publishers: ['lidar_node'],
          subscribers: ['car_controller', 'slam_node'],
          frequency: 40.0,
          last_message: new Date().toISOString(),
          message_count: 6173,
          status: 'active'
        },
        {
          name: '/cmd_vel',
          type: 'geometry_msgs/msg/Twist',
          publishers: ['car_controller'],
          subscribers: ['simulator', 'vesc_driver'],
          frequency: 50.0,
          last_message: new Date().toISOString(),
          message_count: 7716,
          status: 'active'
        },
        {
          name: '/race_monitor/lap_times',
          type: 'race_msgs/msg/LapTime',
          publishers: ['race_monitor'],
          subscribers: ['visualization_engine', 'data_logger'],
          frequency: 0.5,
          last_message: new Date(Date.now() - 5000).toISOString(),
          message_count: 23,
          status: 'slow'
        },
        {
          name: '/race_monitor/position',
          type: 'geometry_msgs/msg/PoseStamped',
          publishers: ['race_monitor'],
          subscribers: ['visualization_engine'],
          frequency: 10.0,
          last_message: new Date().toISOString(),
          message_count: 1543,
          status: 'active'
        },
        {
          name: '/race_monitor/trajectory_analysis',
          type: 'race_msgs/msg/TrajectoryAnalysis',
          publishers: ['trajectory_analyzer'],
          subscribers: ['visualization_engine'],
          frequency: 1.0,
          last_message: new Date(Date.now() - 2000).toISOString(),
          message_count: 156,
          status: 'active'
        },
        {
          name: '/map',
          type: 'nav_msgs/msg/OccupancyGrid',
          publishers: [],
          subscribers: ['visualization_engine'],
          frequency: 0.0,
          last_message: new Date(Date.now() - 60000).toISOString(),
          message_count: 1,
          status: 'inactive'
        },
        {
          name: '/visualization_marker',
          type: 'visualization_msgs/msg/MarkerArray',
          publishers: ['visualization_engine'],
          subscribers: ['rviz'],
          frequency: 5.0,
          last_message: new Date().toISOString(),
          message_count: 772,
          status: 'active'
        }
      ];

      setTopics(mockTopics);
      setLastUpdate(new Date());
    } catch (err) {
      setError('Failed to fetch ROS2 topics');
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
      case 'active':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'slow':
        return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'inactive':
        return 'bg-red-50 text-red-700 border-red-200';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <Activity className="h-4 w-4 text-green-500 animate-pulse" />;
      case 'slow':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'inactive':
        return <Clock className="h-4 w-4 text-red-500" />;
      default:
        return <Activity className="h-4 w-4 text-gray-500" />;
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
    if (freq >= 1) {
      return `${freq.toFixed(1)} Hz`;
    } else if (freq > 0) {
      return `${(freq * 1000).toFixed(0)} mHz`;
    } else {
      return '0 Hz';
    }
  };

  if (loading && topics.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 text-blue-500 animate-spin" />
        <span className="ml-2 text-gray-600">Loading ROS2 topics...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Radio className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">ROS2 Topics</h1>
            <p className="text-gray-600">Monitor ROS2 topic activity and message flow</p>
          </div>
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

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <Activity className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <Radio className="h-5 w-5 text-blue-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Total Topics</p>
              <p className="text-2xl font-bold text-gray-900">{topics.length}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <Activity className="h-5 w-5 text-green-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Active</p>
              <p className="text-2xl font-bold text-green-600">
                {topics.filter(t => t.status === 'active').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <TrendingUp className="h-5 w-5 text-purple-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Avg Frequency</p>
              <p className="text-2xl font-bold text-purple-600">
                {topics.length > 0 ? 
                  formatFrequency(topics.reduce((sum, t) => sum + t.frequency, 0) / topics.length) :
                  '0 Hz'
                }
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <Database className="h-5 w-5 text-orange-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Total Messages</p>
              <p className="text-2xl font-bold text-orange-600">
                {topics.reduce((sum, t) => sum + t.message_count, 0).toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Topics List */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Topic Monitor</h3>
          <p className="text-sm text-gray-600">Click the eye icon to monitor topics in real-time</p>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Topic
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Frequency
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Publishers
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Subscribers
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Messages
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Monitor
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {topics.map((topic, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <div className="flex items-center">
                      {getStatusIcon(topic.status)}
                      <span className="ml-2 text-sm font-medium text-gray-900">
                        {topic.name}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    <code className="bg-gray-100 px-2 py-1 rounded text-xs">
                      {topic.type}
                    </code>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {formatFrequency(topic.frequency)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="space-y-1">
                      {topic.publishers.map((pub, idx) => (
                        <div key={idx} className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                          {pub}
                        </div>
                      ))}
                      {topic.publishers.length === 0 && (
                        <span className="text-xs text-gray-400">None</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="space-y-1">
                      {topic.subscribers.map((sub, idx) => (
                        <div key={idx} className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                          {sub}
                        </div>
                      ))}
                      {topic.subscribers.length === 0 && (
                        <span className="text-xs text-gray-400">None</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {topic.message_count.toLocaleString()}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getStatusColor(topic.status)}`}>
                      {topic.status.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => toggleTopicMonitoring(topic.name)}
                      className={`p-2 rounded-md transition-colors ${
                        monitoringTopics.has(topic.name)
                          ? 'bg-blue-100 text-blue-600 hover:bg-blue-200'
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

      {/* Monitored Topics Detail */}
      {monitoringTopics.size > 0 && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Live Topic Monitor</h3>
            <p className="text-sm text-gray-600">Real-time data from monitored topics</p>
          </div>
          
          <div className="p-4 space-y-4">
            {Array.from(monitoringTopics).map((topicName) => {
              const topic = topics.find(t => t.name === topicName);
              if (!topic) return null;
              
              return (
                <div key={topicName} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-gray-900">{topicName}</h4>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(topic.status)}
                      <span className="text-sm text-gray-600">
                        {formatFrequency(topic.frequency)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded p-3 font-mono text-sm">
                    <div className="text-gray-600">Last message:</div>
                    <div className="text-gray-900">
                      {topic.last_message ? 
                        new Date(topic.last_message).toLocaleString() : 
                        'No recent messages'
                      }
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {topics.length === 0 && !loading && (
        <div className="text-center py-12">
          <Radio className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No ROS2 topics found</h3>
          <p className="text-gray-600">Make sure ROS2 is running and topics are being published.</p>
        </div>
      )}
    </div>
  );
};

export default RosTopics;