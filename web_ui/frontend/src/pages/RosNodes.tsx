import { useState, useEffect } from 'react';
import { 
  Network, 
  CheckCircle, 
  XCircle, 
  RefreshCw, 
  Info,
  Activity,
  Clock,
  Server
} from 'lucide-react';

interface RosNode {
  name: string;
  namespace: string;
  status: 'active' | 'inactive' | 'unknown';
  topics_published: string[];
  topics_subscribed: string[];
  services: string[];
  last_seen?: string;
}

const RosNodes: React.FC = () => {
  const [nodes, setNodes] = useState<RosNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchNodes = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // For now, we'll simulate some ROS2 nodes since the backend might not have full ROS2 integration
      // In a real implementation, this would call `/api/ros/nodes` or similar
      const mockNodes: RosNode[] = [
        {
          name: 'race_monitor',
          namespace: '/race_monitor',
          status: 'active',
          topics_published: ['/race_monitor/lap_times', '/race_monitor/position', '/race_monitor/status'],
          topics_subscribed: ['/odom', '/scan', '/cmd_vel'],
          services: ['/race_monitor/start_race', '/race_monitor/stop_race', '/race_monitor/get_status'],
          last_seen: new Date().toISOString()
        },
        {
          name: 'trajectory_analyzer',
          namespace: '/race_monitor',
          status: 'active',
          topics_published: ['/race_monitor/trajectory_analysis', '/race_monitor/performance_metrics'],
          topics_subscribed: ['/odom', '/race_monitor/lap_times'],
          services: ['/race_monitor/analyze_trajectory'],
          last_seen: new Date().toISOString()
        },
        {
          name: 'visualization_engine',
          namespace: '/race_monitor',
          status: 'active',
          topics_published: ['/visualization_marker', '/race_monitor/viz_data'],
          topics_subscribed: ['/race_monitor/trajectory_analysis', '/race_monitor/position'],
          services: ['/race_monitor/update_visualization'],
          last_seen: new Date().toISOString()
        },
        {
          name: 'car_controller',
          namespace: '/f1tenth',
          status: 'active',
          topics_published: ['/cmd_vel', '/ackermann_cmd'],
          topics_subscribed: ['/scan', '/odom', '/joy'],
          services: ['/f1tenth/emergency_stop', '/f1tenth/reset'],
          last_seen: new Date().toISOString()
        },
        {
          name: 'slam_node',
          namespace: '/slam',
          status: 'inactive',
          topics_published: ['/map', '/pose'],
          topics_subscribed: ['/scan', '/odom'],
          services: ['/slam/save_map', '/slam/reset'],
          last_seen: new Date(Date.now() - 30000).toISOString()
        }
      ];

      setNodes(mockNodes);
      setLastUpdate(new Date());
    } catch (err) {
      setError('Failed to fetch ROS2 nodes');
      console.error('Error fetching nodes:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNodes();
    
    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchNodes, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'inactive':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Activity className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-50 text-green-700 border-green-200';
      case 'inactive':
        return 'bg-red-50 text-red-700 border-red-200';
      default:
        return 'bg-yellow-50 text-yellow-700 border-yellow-200';
    }
  };

  if (loading && nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 text-blue-500 animate-spin" />
        <span className="ml-2 text-gray-600">Loading ROS2 nodes...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Network className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">ROS2 Nodes</h1>
            <p className="text-gray-600">Monitor active ROS2 nodes and their connections</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-500">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
          <button
            onClick={fetchNodes}
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
            <XCircle className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <Server className="h-5 w-5 text-blue-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Total Nodes</p>
              <p className="text-2xl font-bold text-gray-900">{nodes.length}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Active</p>
              <p className="text-2xl font-bold text-green-600">
                {nodes.filter(n => n.status === 'active').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <XCircle className="h-5 w-5 text-red-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Inactive</p>
              <p className="text-2xl font-bold text-red-600">
                {nodes.filter(n => n.status === 'inactive').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center">
            <Activity className="h-5 w-5 text-purple-600 mr-2" />
            <div>
              <p className="text-sm text-gray-600">Race Nodes</p>
              <p className="text-2xl font-bold text-purple-600">
                {nodes.filter(n => n.namespace.includes('race')).length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Nodes List */}
      <div className="space-y-4">
        {nodes.map((node, index) => (
          <div key={index} className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                {getStatusIcon(node.status)}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{node.name}</h3>
                  <p className="text-sm text-gray-600">{node.namespace}</p>
                </div>
              </div>
              
              <div className={`px-3 py-1 rounded-full text-sm font-medium border ${getStatusColor(node.status)}`}>
                {node.status.toUpperCase()}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-6">
              {/* Published Topics */}
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
                  <Info className="h-4 w-4 mr-1" />
                  Published Topics ({node.topics_published.length})
                </h4>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {node.topics_published.map((topic, idx) => (
                    <div key={idx} className="text-sm text-gray-600 bg-gray-50 px-2 py-1 rounded">
                      {topic}
                    </div>
                  ))}
                </div>
              </div>

              {/* Subscribed Topics */}
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
                  <Info className="h-4 w-4 mr-1" />
                  Subscribed Topics ({node.topics_subscribed.length})
                </h4>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {node.topics_subscribed.map((topic, idx) => (
                    <div key={idx} className="text-sm text-gray-600 bg-gray-50 px-2 py-1 rounded">
                      {topic}
                    </div>
                  ))}
                </div>
              </div>

              {/* Services */}
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
                  <Info className="h-4 w-4 mr-1" />
                  Services ({node.services.length})
                </h4>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {node.services.map((service, idx) => (
                    <div key={idx} className="text-sm text-gray-600 bg-gray-50 px-2 py-1 rounded">
                      {service}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {node.last_seen && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex items-center text-sm text-gray-500">
                  <Clock className="h-4 w-4 mr-1" />
                  Last seen: {new Date(node.last_seen).toLocaleString()}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {nodes.length === 0 && !loading && (
        <div className="text-center py-12">
          <Network className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No ROS2 nodes found</h3>
          <p className="text-gray-600">Make sure ROS2 is running and nodes are active.</p>
        </div>
      )}
    </div>
  );
};

export default RosNodes;