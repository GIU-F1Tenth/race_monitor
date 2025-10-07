import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Settings, 
  BarChart3, 
  Activity, 
  Monitor,
  Gauge,
  Network,
  Radio,
  ChevronLeft,
  ChevronRight,
  Circle,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import ConfigEditor from './ConfigEditor';

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

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  rosStatus: RosStatus;
  notificationCounts?: NotificationCounts;
}

const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggle, rosStatus, notificationCounts }) => {
  const location = useLocation();

  const navItems = [
    { 
      path: '/', 
      label: 'Dashboard', 
      icon: Home,
      group: 'main'
    },
    { 
      path: '/live', 
      label: 'Live Monitor', 
      icon: Monitor,
      group: 'main'
    },
    { 
      path: '/results', 
      label: 'Race Results', 
      icon: BarChart3,
      group: 'race',
      badge: notificationCounts?.unread_results
    },
    { 
      path: '/analysis', 
      label: 'EVO Analysis', 
      icon: Activity,
      group: 'race',
      badge: notificationCounts?.unread_analysis
    },
    { 
      path: '/ros-nodes', 
      label: 'ROS2 Nodes', 
      icon: Network,
      group: 'ros',
      badge: rosStatus.active_nodes
    },
    { 
      path: '/ros-topics', 
      label: 'ROS2 Topics', 
      icon: Radio,
      group: 'ros',
      badge: rosStatus.active_topics
    },
    { 
      path: '/config', 
      label: 'Configuration', 
      icon: Settings,
      group: 'config'
    },
  ];

  const getGroupLabel = (group: string) => {
    switch (group) {
      case 'main': return 'Main';
      case 'race': return 'Race Data';
      case 'ros': return 'ROS2 System';
      case 'config': return 'Settings';
      default: return '';
    }
  };

  const getStatusIcon = () => {
    if (!rosStatus.ros_available) {
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    }
    if (rosStatus.monitoring_active) {
      return <CheckCircle2 className="h-4 w-4 text-green-500 animate-pulse" />;
    }
    return <Circle className="h-4 w-4 text-yellow-500" />;
  };

  const getStatusText = () => {
    if (!rosStatus.ros_available) {
      return 'ROS2 Offline';
    }
    if (rosStatus.monitoring_active) {
      return 'Monitoring Active';
    }
    return 'ROS2 Ready';
  };

  const getStatusColor = () => {
    if (!rosStatus.ros_available) {
      return 'text-red-600';
    }
    if (rosStatus.monitoring_active) {
      return 'text-green-600';
    }
    return 'text-yellow-600';
  };

  const groupedItems = navItems.reduce((acc, item) => {
    if (!acc[item.group]) {
      acc[item.group] = [];
    }
    acc[item.group].push(item);
    return acc;
  }, {} as Record<string, typeof navItems>);

  return (
    <div className={`fixed left-0 top-0 h-full bg-white shadow-lg border-r border-gray-200 z-50 transition-all duration-300 ${
      collapsed ? 'w-16' : 'w-64'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className={`flex items-center space-x-3 transition-opacity duration-200 ${
          collapsed ? 'opacity-0' : 'opacity-100'
        }`}>
          <Gauge className="h-8 w-8 text-blue-600" />
          <div>
            <h1 className="text-lg font-bold text-gray-900">Race Monitor</h1>
            <p className="text-xs text-gray-500">ROS2 F1Tenth</p>
          </div>
        </div>
        
        <button
          onClick={onToggle}
          className="p-1 rounded-md hover:bg-gray-100 transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="h-5 w-5 text-gray-600" />
          ) : (
            <ChevronLeft className="h-5 w-5 text-gray-600" />
          )}
        </button>
      </div>

      {/* ROS2 Status */}
      <div className="p-4 border-b border-gray-200">
        <div className={`flex items-center space-x-3 ${collapsed ? 'justify-center' : ''}`}>
          {getStatusIcon()}
          {!collapsed && (
            <div>
              <p className={`text-sm font-medium ${getStatusColor()}`}>
                {getStatusText()}
              </p>
              <p className="text-xs text-gray-500">
                Last update: {new Date(rosStatus.timestamp).toLocaleTimeString()}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-6 overflow-y-auto">
        {Object.entries(groupedItems).map(([group, items]) => (
          <div key={group}>
            {!collapsed && (
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                {getGroupLabel(group)}
              </h3>
            )}
            <div className="space-y-1">
              {items.map(({ path, label, icon: Icon, badge }) => (
                <Link
                  key={path}
                  to={path}
                  className={`
                    flex items-center justify-between px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 relative
                    ${location.pathname === path
                      ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }
                  `}
                  title={collapsed ? label : ''}
                >
                  <div className={`flex items-center space-x-3 ${collapsed ? 'justify-center' : ''}`}>
                    <Icon className={`h-5 w-5 ${collapsed ? '' : 'flex-shrink-0'}`} />
                    {!collapsed && <span>{label}</span>}
                  </div>
                  
                  {!collapsed && badge !== undefined && badge > 0 && (
                    <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                      (path === '/results' || path === '/analysis') && notificationCounts
                        ? 'bg-red-100 text-red-700'
                        : 'bg-blue-100 text-blue-700'
                    }`}>
                      {badge}
                    </span>
                  )}
                  
                  {collapsed && badge !== undefined && badge > 0 && (
                    <div className={`absolute -top-1 -right-1 h-5 w-5 rounded-full text-xs font-bold flex items-center justify-center ${
                      (path === '/results' || path === '/analysis') && notificationCounts
                        ? 'bg-red-500 text-white'
                        : 'bg-blue-500 text-white'
                    }`}>
                      {badge > 99 ? '99+' : badge}
                    </div>
                  )}
                </Link>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Config Editor */}
      <ConfigEditor isCollapsed={collapsed} />

      {/* Footer */}
      {!collapsed && (
        <div className="p-4 border-t border-gray-200">
          <div className="text-xs text-gray-500 text-center">
            <p>ROS2 Race Monitor</p>
            <p>v1.0.0</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;