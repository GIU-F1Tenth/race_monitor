import { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Download, 
  Settings,
  RefreshCw,
  Eye,
  EyeOff
} from 'lucide-react';

interface Node {
  id: string;
  name: string;
  type: 'publisher' | 'subscriber' | 'service' | 'action';
  x: number;
  y: number;
  connections: string[];
}

interface Topic {
  name: string;
  type: string;
  publishers: string[];
  subscribers: string[];
}

const RqtGraph: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [showTopics, setShowTopics] = useState(true);
  const [showServices, setShowServices] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchGraphData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Add timeout to prevent infinite loading
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const [nodesResponse, topicsResponse] = await Promise.all([
        fetch('/api/ros/nodes', { signal: controller.signal }),
        fetch('/api/ros/topics', { signal: controller.signal })
      ]);

      clearTimeout(timeoutId);

      if (nodesResponse.ok && topicsResponse.ok) {
        const nodesData = await nodesResponse.json();
        const topicsData = await topicsResponse.json();
        
        // Convert ROS data to graph format
        const graphNodes: Node[] = (nodesData.nodes || []).map((node: any, index: number) => ({
          id: node.name || `node_${index}`,
          name: node.name || `Unknown Node ${index}`,
          type: node.topics_published?.length > 0 ? 'publisher' : 'subscriber',
          x: 100 + (index % 5) * 150,
          y: 100 + Math.floor(index / 5) * 150,
          connections: [...(node.topics_published || []), ...(node.topics_subscribed || [])]
        }));

        const graphTopics: Topic[] = (topicsData.topics || []).map((topic: any) => ({
          name: topic.name || 'Unknown Topic',
          type: topic.type || 'unknown',
          publishers: topic.publishers || [],
          subscribers: topic.subscribers || []
        }));

        setNodes(graphNodes);
        setTopics(graphTopics);
      } else {
        // If APIs not available, show empty state
        setNodes([]);
        setTopics([]);
        setError('ROS2 graph data not available. Make sure ROS2 is running and backend is connected.');
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please check your connection.');
      } else {
        setError('Failed to connect to ROS2 system. Check if ROS2 is running.');
      }
      setNodes([]);
      setTopics([]);
      console.error('Error fetching graph data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraphData();
  }, []);

  useEffect(() => {
    drawGraph();
  }, [nodes, topics, selectedNode, zoom, offset, showTopics, showServices]);

  const drawGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply zoom and offset
    ctx.save();
    ctx.scale(zoom, zoom);
    ctx.translate(offset.x, offset.y);

    // Draw connections
    topics.forEach(topic => {
      if (!showTopics && topic.type.includes('msg')) return;
      if (!showServices && topic.type.includes('srv')) return;

      topic.publishers.forEach(pubId => {
        topic.subscribers.forEach(subId => {
          const pubNode = nodes.find(n => n.name === `/${pubId}` || n.id === pubId);
          const subNode = nodes.find(n => n.name === `/${subId}` || n.id === subId);
          
          if (pubNode && subNode) {
            ctx.beginPath();
            ctx.moveTo(pubNode.x, pubNode.y);
            ctx.lineTo(subNode.x, subNode.y);
            ctx.strokeStyle = selectedNode === pubNode.id || selectedNode === subNode.id 
              ? '#3B82F6' : '#E5E7EB';
            ctx.lineWidth = selectedNode === pubNode.id || selectedNode === subNode.id ? 2 : 1;
            ctx.stroke();

            // Draw arrow
            const angle = Math.atan2(subNode.y - pubNode.y, subNode.x - pubNode.x);
            const arrowLength = 10;
            ctx.beginPath();
            ctx.moveTo(subNode.x, subNode.y);
            ctx.lineTo(
              subNode.x - arrowLength * Math.cos(angle - Math.PI / 6),
              subNode.y - arrowLength * Math.sin(angle - Math.PI / 6)
            );
            ctx.moveTo(subNode.x, subNode.y);
            ctx.lineTo(
              subNode.x - arrowLength * Math.cos(angle + Math.PI / 6),
              subNode.y - arrowLength * Math.sin(angle + Math.PI / 6)
            );
            ctx.stroke();
          }
        });
      });
    });

    // Draw nodes
    nodes.forEach(node => {
      const isSelected = selectedNode === node.id;
      const isVisible = searchTerm === '' || 
        node.name.toLowerCase().includes(searchTerm.toLowerCase());
      
      if (!isVisible) return;

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, isSelected ? 25 : 20, 0, 2 * Math.PI);
      
      // Node color based on type
      switch (node.type) {
        case 'publisher':
          ctx.fillStyle = isSelected ? '#10B981' : '#D1FAE5';
          ctx.strokeStyle = '#10B981';
          break;
        case 'subscriber':
          ctx.fillStyle = isSelected ? '#3B82F6' : '#DBEAFE';
          ctx.strokeStyle = '#3B82F6';
          break;
        case 'service':
          ctx.fillStyle = isSelected ? '#F59E0B' : '#FEF3C7';
          ctx.strokeStyle = '#F59E0B';
          break;
        default:
          ctx.fillStyle = isSelected ? '#6B7280' : '#F3F4F6';
          ctx.strokeStyle = '#6B7280';
      }
      
      ctx.fill();
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.stroke();

      // Node label
      ctx.fillStyle = '#1F2937';
      ctx.font = isSelected ? 'bold 12px Arial' : '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(node.name, node.x, node.y - 30);
    });

    ctx.restore();
  };

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left - offset.x * zoom) / zoom;
    const y = (event.clientY - rect.top - offset.y * zoom) / zoom;

    // Check if click is on a node
    const clickedNode = nodes.find(node => {
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
      return distance <= 25;
    });

    setSelectedNode(clickedNode ? clickedNode.id : null);
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      const deltaX = event.clientX - lastMousePos.x;
      const deltaY = event.clientY - lastMousePos.y;
      
      setOffset(prev => ({
        x: prev.x + deltaX / zoom,
        y: prev.y + deltaY / zoom
      }));
      
      setLastMousePos({ x: event.clientX, y: event.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setSelectedNode(null);
  };

  const exportGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = 'rqt_graph.png';
    link.href = canvas.toDataURL();
    link.click();
  };

  const refreshGraph = async () => {
    setLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setLoading(false);
  };

  const filteredNodes = nodes.filter(node => 
    searchTerm === '' || node.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const selectedNodeData = selectedNode ? nodes.find(n => n.id === selectedNode) : null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">RQT Graph</h1>
          <p className="text-gray-600">ROS2 Node and Topic Visualization</p>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={refreshGraph}
            disabled={loading}
            className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search nodes..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowTopics(!showTopics)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium flex items-center space-x-1 ${
                    showTopics ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {showTopics ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                  <span>Topics</span>
                </button>
                
                <button
                  onClick={() => setShowServices(!showServices)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium flex items-center space-x-1 ${
                    showServices ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {showServices ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                  <span>Services</span>
                </button>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
                className="p-2 rounded-lg hover:bg-gray-100"
                title="Zoom Out"
              >
                <ZoomOut className="h-4 w-4 text-gray-600" />
              </button>
              
              <span className="text-sm text-gray-600 min-w-[60px] text-center">
                {Math.round(zoom * 100)}%
              </span>
              
              <button
                onClick={() => setZoom(Math.min(2, zoom + 0.1))}
                className="p-2 rounded-lg hover:bg-gray-100"
                title="Zoom In"
              >
                <ZoomIn className="h-4 w-4 text-gray-600" />
              </button>
              
              <button
                onClick={resetView}
                className="p-2 rounded-lg hover:bg-gray-100"
                title="Reset View"
              >
                <RotateCcw className="h-4 w-4 text-gray-600" />
              </button>
              
              <button
                onClick={exportGraph}
                className="p-2 rounded-lg hover:bg-gray-100"
                title="Export"
              >
                <Download className="h-4 w-4 text-gray-600" />
              </button>
            </div>
          </div>
        </div>
        
        <div className="flex">
          <div className="flex-1">
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className="border-r border-gray-200 cursor-move"
              onClick={handleCanvasClick}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
          </div>
          
          <div className="w-80 p-4 bg-gray-50">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Node Information</h3>
            
            {selectedNodeData ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-900">{selectedNodeData.name}</h4>
                  <span className={`inline-block px-2 py-1 text-xs rounded-full font-medium mt-1 ${
                    selectedNodeData.type === 'publisher' ? 'bg-green-100 text-green-700' :
                    selectedNodeData.type === 'subscriber' ? 'bg-blue-100 text-blue-700' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>
                    {selectedNodeData.type}
                  </span>
                </div>
                
                <div>
                  <h5 className="font-medium text-gray-700 mb-2">Connections</h5>
                  <div className="space-y-1">
                    {selectedNodeData.connections.map((connection, index) => (
                      <div key={index} className="text-sm text-gray-600 bg-white p-2 rounded border">
                        {connection}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500">
                <Settings className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                <p>Click on a node to see details</p>
              </div>
            )}
            
            <div className="mt-6">
              <h5 className="font-medium text-gray-700 mb-2">Legend</h5>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-green-200 border-2 border-green-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">Publisher</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-blue-200 border-2 border-blue-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">Subscriber</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-yellow-200 border-2 border-yellow-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">Service</span>
                </div>
              </div>
            </div>
            
            <div className="mt-6">
              <h5 className="font-medium text-gray-700 mb-2">Statistics</h5>
              <div className="text-sm text-gray-600 space-y-1">
                <p>Total Nodes: {filteredNodes.length}</p>
                <p>Total Topics: {topics.length}</p>
                <p>Publishers: {filteredNodes.filter(n => n.type === 'publisher').length}</p>
                <p>Subscribers: {filteredNodes.filter(n => n.type === 'subscriber').length}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RqtGraph;