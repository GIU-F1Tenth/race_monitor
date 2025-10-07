import { useState, useEffect, useRef } from 'react';
import { 
  Network, 
  RefreshCw, 
  Download, 
  ZoomIn,
  ZoomOut,
  Search
} from 'lucide-react';

interface RosNode {
  name: string;
  namespace: string;
  x: number;
  y: number;
  type: 'publisher' | 'subscriber' | 'service' | 'node';
}

interface RosConnection {
  from: string;
  to: string;
  topic: string;
  type: 'topic' | 'service';
}

interface RQTGraphProps {
  realtime?: boolean;
  height?: number;
}

const RQTGraph: React.FC<RQTGraphProps> = ({ realtime = false, height = 400 }) => {
  const [nodes, setNodes] = useState<RosNode[]>([]);
  const [connections, setConnections] = useState<RosConnection[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchFilter, setSearchFilter] = useState('');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchGraphData = async () => {
    setLoading(true);
    try {
      // Mock RQT graph data - in real implementation, this would come from ROS2 graph discovery
      const mockNodes: RosNode[] = [
        { name: 'race_monitor', namespace: '/race_monitor', x: 200, y: 150, type: 'node' },
        { name: 'trajectory_analyzer', namespace: '/race_monitor', x: 400, y: 100, type: 'node' },
        { name: 'visualization_engine', namespace: '/race_monitor', x: 400, y: 200, type: 'node' },
        { name: 'car_controller', namespace: '/f1tenth', x: 100, y: 250, type: 'node' },
        { name: 'lidar_node', namespace: '/sensors', x: 50, y: 100, type: 'node' },
        { name: 'slam_node', namespace: '/slam', x: 300, y: 300, type: 'node' },
        { name: 'rviz', namespace: '/viz', x: 500, y: 250, type: 'node' }
      ];

      const mockConnections: RosConnection[] = [
        { from: 'lidar_node', to: 'car_controller', topic: '/scan', type: 'topic' },
        { from: 'lidar_node', to: 'slam_node', topic: '/scan', type: 'topic' },
        { from: 'car_controller', to: 'race_monitor', topic: '/odom', type: 'topic' },
        { from: 'car_controller', to: 'trajectory_analyzer', topic: '/odom', type: 'topic' },
        { from: 'race_monitor', to: 'visualization_engine', topic: '/race_monitor/lap_times', type: 'topic' },
        { from: 'race_monitor', to: 'visualization_engine', topic: '/race_monitor/position', type: 'topic' },
        { from: 'trajectory_analyzer', to: 'visualization_engine', topic: '/race_monitor/trajectory_analysis', type: 'topic' },
        { from: 'visualization_engine', to: 'rviz', topic: '/visualization_marker', type: 'topic' },
        { from: 'slam_node', to: 'visualization_engine', topic: '/map', type: 'topic' }
      ];

      setNodes(mockNodes);
      setConnections(mockConnections);
    } catch (error) {
      console.error('Failed to fetch RQT graph data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraphData();
    
    if (realtime) {
      const interval = setInterval(fetchGraphData, 5000); // Update every 5 seconds
      return () => clearInterval(interval);
    }
  }, [realtime]);

  useEffect(() => {
    drawGraph();
  }, [nodes, connections, searchFilter, selectedNode, zoomLevel]);

  const drawGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height: canvasHeight } = canvas;
    ctx.clearRect(0, 0, width, canvasHeight);

    // Apply zoom and filter
    ctx.save();
    ctx.scale(zoomLevel, zoomLevel);

    const filteredNodes = nodes.filter(node => 
      searchFilter === '' || 
      node.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
      node.namespace.toLowerCase().includes(searchFilter.toLowerCase())
    );

    // Draw connections first (so they appear behind nodes)
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    connections.forEach(connection => {
      const fromNode = nodes.find(n => n.name === connection.from);
      const toNode = nodes.find(n => n.name === connection.to);
      
      if (fromNode && toNode) {
        // Only draw if both nodes are visible
        const fromVisible = filteredNodes.includes(fromNode);
        const toVisible = filteredNodes.includes(toNode);
        
        if (fromVisible && toVisible) {
          ctx.beginPath();
          ctx.moveTo(fromNode.x + 50, fromNode.y + 25); // Center of node
          ctx.lineTo(toNode.x + 50, toNode.y + 25);
          
          // Color code by connection type
          ctx.strokeStyle = connection.type === 'topic' ? '#3b82f6' : '#10b981';
          ctx.stroke();
          
          // Draw arrow
          const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x);
          const arrowLength = 10;
          const arrowAngle = Math.PI / 6;
          
          ctx.beginPath();
          ctx.moveTo(toNode.x + 40, toNode.y + 25);
          ctx.lineTo(
            toNode.x + 40 - arrowLength * Math.cos(angle - arrowAngle),
            toNode.y + 25 - arrowLength * Math.sin(angle - arrowAngle)
          );
          ctx.moveTo(toNode.x + 40, toNode.y + 25);
          ctx.lineTo(
            toNode.x + 40 - arrowLength * Math.cos(angle + arrowAngle),
            toNode.y + 25 - arrowLength * Math.sin(angle + arrowAngle)
          );
          ctx.stroke();
          
          // Draw topic label
          const midX = (fromNode.x + toNode.x) / 2 + 50;
          const midY = (fromNode.y + toNode.y) / 2 + 25;
          ctx.fillStyle = '#374151';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(connection.topic, midX, midY - 5);
        }
      }
    });

    // Draw nodes
    filteredNodes.forEach(node => {
      const isSelected = selectedNode === node.name;
      const isHighlighted = searchFilter !== '' && 
        (node.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
         node.namespace.toLowerCase().includes(searchFilter.toLowerCase()));

      // Node background
      ctx.fillStyle = isSelected ? '#dbeafe' : isHighlighted ? '#fef3c7' : '#f9fafb';
      ctx.strokeStyle = isSelected ? '#3b82f6' : '#d1d5db';
      ctx.lineWidth = isSelected ? 3 : 1;
      
      const nodeWidth = 100;
      const nodeHeight = 50;
      
      ctx.fillRect(node.x, node.y, nodeWidth, nodeHeight);
      ctx.strokeRect(node.x, node.y, nodeWidth, nodeHeight);
      
      // Node icon (simple circle)
      ctx.fillStyle = getNodeColor(node.type);
      ctx.beginPath();
      ctx.arc(node.x + 15, node.y + 15, 8, 0, 2 * Math.PI);
      ctx.fill();
      
      // Node text
      ctx.fillStyle = '#374151';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(node.name, node.x + 28, node.y + 12);
      
      ctx.font = '8px sans-serif';
      ctx.fillStyle = '#6b7280';
      ctx.fillText(node.namespace, node.x + 28, node.y + 25);
      
      // Node type
      ctx.fillText(node.type, node.x + 5, node.y + 45);
    });

    ctx.restore();
  };

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'node': return '#3b82f6';
      case 'publisher': return '#10b981';
      case 'subscriber': return '#f59e0b';
      case 'service': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / zoomLevel;
    const y = (event.clientY - rect.top) / zoomLevel;

    // Check if click is on a node
    const clickedNode = nodes.find(node => 
      x >= node.x && x <= node.x + 100 &&
      y >= node.y && y <= node.y + 50
    );

    setSelectedNode(clickedNode ? clickedNode.name : null);
  };

  const exportGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = `rqt_graph_${new Date().toISOString().split('T')[0]}.png`;
    link.href = canvas.toDataURL();
    link.click();
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Network className="h-6 w-6 text-blue-600" />
          <div>
            <h3 className="text-lg font-semibold text-gray-900">RQT Graph - ROS2 Node Graph</h3>
            <p className="text-sm text-gray-600">
              {realtime ? 'Real-time node and topic visualization' : 'Static node graph view'}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Filter nodes..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <button
            onClick={() => setZoomLevel(prev => Math.max(0.5, prev - 0.1))}
            className="p-2 rounded-md hover:bg-gray-100"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4 text-gray-600" />
          </button>
          
          <button
            onClick={() => setZoomLevel(prev => Math.min(2, prev + 0.1))}
            className="p-2 rounded-md hover:bg-gray-100"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4 text-gray-600" />
          </button>
          
          <button
            onClick={exportGraph}
            className="p-2 rounded-md hover:bg-gray-100"
            title="Export graph"
          >
            <Download className="h-4 w-4 text-gray-600" />
          </button>
          
          <button
            onClick={fetchGraphData}
            disabled={loading}
            className="p-2 rounded-md hover:bg-gray-100"
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Graph Canvas */}
      <div ref={containerRef} className="relative bg-gray-50 rounded-lg p-4 overflow-auto">
        <canvas
          ref={canvasRef}
          width={800}
          height={height}
          className="border border-gray-200 rounded cursor-pointer"
          onClick={handleCanvasClick}
        />
        
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 rounded-lg">
            <RefreshCw className="h-8 w-8 text-blue-500 animate-spin" />
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <div className="text-sm text-gray-600">
            <span className="font-medium">Zoom:</span> {(zoomLevel * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-gray-600">
            <span className="font-medium">Nodes:</span> {nodes.length}
          </div>
          <div className="text-sm text-gray-600">
            <span className="font-medium">Connections:</span> {connections.length}
          </div>
        </div>
        
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-blue-600 rounded-full"></div>
            <span>Nodes</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-1 bg-blue-600"></div>
            <span>Topics</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-1 bg-green-600"></div>
            <span>Services</span>
          </div>
        </div>
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Selected Node: {selectedNode}</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-blue-800">Publishes to:</span>
              <ul className="mt-1 space-y-1">
                {connections
                  .filter(c => c.from === selectedNode)
                  .map((c, idx) => (
                    <li key={idx} className="text-blue-700">• {c.topic}</li>
                  ))
                }
              </ul>
            </div>
            <div>
              <span className="font-medium text-blue-800">Subscribes to:</span>
              <ul className="mt-1 space-y-1">
                {connections
                  .filter(c => c.to === selectedNode)
                  .map((c, idx) => (
                    <li key={idx} className="text-blue-700">• {c.topic}</li>
                  ))
                }
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RQTGraph;