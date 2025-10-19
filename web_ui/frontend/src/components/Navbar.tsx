import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Home, 
  Settings, 
  BarChart3, 
  Activity, 
  Monitor,
  Gauge
} from 'lucide-react';

const Navbar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/config', label: 'Configuration', icon: Settings },
    { path: '/results', label: 'Results', icon: BarChart3 },
    { path: '/analysis', label: 'Analysis', icon: Activity },
    { path: '/live', label: 'Live Monitor', icon: Monitor },
  ];

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-2">
            <Gauge className="h-8 w-8 text-race-primary" />
            <span className="text-xl font-bold text-gray-900">Race Monitor</span>
          </div>

          {/* Navigation Links */}
          <div className="flex space-x-1">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`
                  flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200
                  ${location.pathname === path
                    ? 'bg-race-primary text-white'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }
                `}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </Link>
            ))}
          </div>

          {/* Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600">Online</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;