import { useState, useEffect } from 'react';
import { 
  FileText, 
  Plus, 
  Trash2, 
  Save, 
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Edit,
  Pin,
  Settings
} from 'lucide-react';

interface ConfigParam {
  key: string;
  value: any;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  comment?: string;
}

interface ConfigFile {
  filename: string;
  content: string;
  params: ConfigParam[];
  modified: boolean;
  isPinned: boolean;
}

interface ConfigEditorProps {
  isCollapsed: boolean;
}

const ConfigEditor: React.FC<ConfigEditorProps> = ({ isCollapsed }) => {
  const [configFiles, setConfigFiles] = useState<ConfigFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('race_monitor.yaml');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newFileName, setNewFileName] = useState('');
  const [showNewFile, setShowNewFile] = useState(false);
  const [editMode, setEditMode] = useState<'visual' | 'raw'>('visual');

  // Parse YAML-like content to extract parameters
  const parseYamlContent = (content: string): ConfigParam[] => {
    const params: ConfigParam[] = [];
    const lines = content.split('\n');
    
    let currentSection = '';
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Skip comments and empty lines
      if (trimmed.startsWith('#') || trimmed === '') continue;
      
      // Check for ros__parameters section
      if (trimmed.includes('ros__parameters:')) {
        currentSection = 'ros__parameters';
        continue;
      }
      
      // Parse parameter lines
      if (currentSection === 'ros__parameters' && trimmed.includes(':')) {
        const [key, ...valueParts] = trimmed.split(':');
        const value = valueParts.join(':').trim();
        
        if (key && value) {
          let parsedValue: any = value;
          let type: ConfigParam['type'] = 'string';
          
          // Try to parse the value type
          if (value === 'true' || value === 'false') {
            parsedValue = value === 'true';
            type = 'boolean';
          } else if (!isNaN(Number(value)) && value !== '') {
            parsedValue = Number(value);
            type = 'number';
          } else if (value.startsWith('[') && value.endsWith(']')) {
            try {
              parsedValue = JSON.parse(value.replace(/'/g, '"'));
              type = 'array';
            } catch {
              parsedValue = value;
              type = 'string';
            }
          }
          
          params.push({
            key: key.trim(),
            value: parsedValue,
            type,
            comment: ''
          });
        }
      }
    }
    
    return params;
  };

  // Convert parameters back to YAML format
  const paramsToYaml = (params: ConfigParam[], filename: string): string => {
    const header = `# ${filename}
# Configuration file for race monitor

race_monitor:
  ros__parameters:`;
    
    const paramLines = params.map(param => {
      let valueStr = param.value;
      if (param.type === 'array') {
        valueStr = JSON.stringify(param.value);
      } else if (param.type === 'boolean') {
        valueStr = param.value ? 'true' : 'false';
      }
      
      const comment = param.comment ? ` # ${param.comment}` : '';
      return `    ${param.key}: ${valueStr}${comment}`;
    }).join('\n');
    
    return `${header}\n${paramLines}\n`;
  };

  // Load available config files
  const fetchConfigFiles = async () => {
    setLoading(true);
    try {
      // Add timeout to prevent infinite loading
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch('/api/config/list', { signal: controller.signal });
      
      if (response.ok) {
        const files = await response.json();
        const configFilesData: ConfigFile[] = [];
        
        // Always include race_monitor.yaml first (pinned)
        if (!files.includes('race_monitor.yaml')) {
          files.unshift('race_monitor.yaml');
        }
        
        for (const filename of files) {
          try {
            const contentResponse = await fetch(`/api/config/${filename}`, { signal: controller.signal });
            if (contentResponse.ok) {
              const content = await contentResponse.text();
              const params = parseYamlContent(content);
              
              configFilesData.push({
                filename,
                content,
                params,
                modified: false,
                isPinned: filename === 'race_monitor.yaml'
              });
            }
          } catch (err) {
            console.error(`Failed to load ${filename}:`, err);
          }
        }
        
        setConfigFiles(configFilesData);
        setError(null);
        clearTimeout(timeoutId);
      } else {
        // If config API not available, show error instead of infinite loading
        setConfigFiles([]);
        setError('Configuration API not available. Backend may not be running.');
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please check your connection.');
      } else {
        setError('Failed to fetch config files. Backend may not be running.');
      }
      setConfigFiles([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigFiles();
  }, []);

  const handleFileSelect = (filename: string) => {
    setSelectedFile(filename);
  };

  const handleParamChange = (filename: string, paramKey: string, newValue: any) => {
    setConfigFiles(prev => 
      prev.map(file => {
        if (file.filename === filename) {
          const updatedParams = file.params.map(param => 
            param.key === paramKey ? { ...param, value: newValue } : param
          );
          const updatedContent = paramsToYaml(updatedParams, filename);
          
          return {
            ...file,
            params: updatedParams,
            content: updatedContent,
            modified: true
          };
        }
        return file;
      })
    );
  };

  const handleRawContentChange = (filename: string, newContent: string) => {
    setConfigFiles(prev => 
      prev.map(file => 
        file.filename === filename 
          ? { 
              ...file, 
              content: newContent, 
              params: parseYamlContent(newContent),
              modified: true 
            }
          : file
      )
    );
  };

  const handleAddParam = (filename: string) => {
    const newParam: ConfigParam = {
      key: 'new_param',
      value: '',
      type: 'string',
      comment: ''
    };
    
    setConfigFiles(prev => 
      prev.map(file => {
        if (file.filename === filename) {
          const updatedParams = [...file.params, newParam];
          const updatedContent = paramsToYaml(updatedParams, filename);
          
          return {
            ...file,
            params: updatedParams,
            content: updatedContent,
            modified: true
          };
        }
        return file;
      })
    );
  };

  const handleRemoveParam = (filename: string, paramKey: string) => {
    setConfigFiles(prev => 
      prev.map(file => {
        if (file.filename === filename) {
          const updatedParams = file.params.filter(param => param.key !== paramKey);
          const updatedContent = paramsToYaml(updatedParams, filename);
          
          return {
            ...file,
            params: updatedParams,
            content: updatedContent,
            modified: true
          };
        }
        return file;
      })
    );
  };

  const handleSaveFile = async (filename: string) => {
    const file = configFiles.find(f => f.filename === filename);
    if (!file) return;

    try {
      const response = await fetch(`/api/config/${filename}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: file.content }),
      });

      if (response.ok) {
        setConfigFiles(prev => 
          prev.map(f => 
            f.filename === filename 
              ? { ...f, modified: false }
              : f
          )
        );
      } else {
        setError(`Failed to save ${filename}`);
      }
    } catch (err) {
      setError(`Error saving ${filename}`);
    }
  };

  const handleDeleteFile = async (filename: string) => {
    if (filename === 'race_monitor.yaml') {
      setError('Cannot delete race_monitor.yaml - it is required');
      return;
    }

    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/config/${filename}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setConfigFiles(prev => prev.filter(f => f.filename !== filename));
        if (selectedFile === filename) {
          setSelectedFile('race_monitor.yaml');
        }
      } else {
        setError(`Failed to delete ${filename}`);
      }
    } catch (err) {
      setError(`Error deleting ${filename}`);
    }
  };

  const handleCreateFile = async () => {
    if (!newFileName.trim()) {
      setError('Please enter a filename');
      return;
    }

    const filename = newFileName.endsWith('.yaml') ? newFileName : `${newFileName}.yaml`;
    
    if (configFiles.some(f => f.filename === filename)) {
      setError('File already exists');
      return;
    }

    const defaultContent = `# ${filename}
# Configuration file for race monitor

race_monitor:
  ros__parameters:
    # Add your parameters here
    example_param: "example_value"
    enable_feature: true
    max_speed: 5.0
`;

    try {
      const response = await fetch(`/api/config/${filename}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: defaultContent }),
      });

      if (response.ok) {
        setConfigFiles(prev => [...prev, {
          filename,
          content: defaultContent,
          params: parseYamlContent(defaultContent),
          modified: false,
          isPinned: false
        }]);
        setSelectedFile(filename);
        setNewFileName('');
        setShowNewFile(false);
        setError(null);
      } else {
        setError(`Failed to create ${filename}`);
      }
    } catch (err) {
      setError(`Error creating ${filename}`);
    }
  };

  const selectedFileData = configFiles.find(f => f.filename === selectedFile);

  const renderParamInput = (param: ConfigParam, filename: string) => {
    switch (param.type) {
      case 'boolean':
        return (
          <select
            value={param.value ? 'true' : 'false'}
            onChange={(e) => handleParamChange(filename, param.key, e.target.value === 'true')}
            className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        );
      case 'number':
        return (
          <input
            type="number"
            value={param.value}
            step="any"
            onChange={(e) => handleParamChange(filename, param.key, Number(e.target.value))}
            className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        );
      default:
        return (
          <input
            type="text"
            value={typeof param.value === 'string' ? param.value : JSON.stringify(param.value)}
            onChange={(e) => handleParamChange(filename, param.key, e.target.value)}
            className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        );
    }
  };

  if (isCollapsed) {
    return (
      <div className="p-2 border-t border-gray-200">
        <div className="flex justify-center">
          <Settings className="h-5 w-5 text-gray-600" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 border-t border-gray-200">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900">Config Editor</h3>
        <div className="flex space-x-1">
          <button
            onClick={fetchConfigFiles}
            disabled={loading}
            className="p-1 rounded hover:bg-gray-100"
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={() => setShowNewFile(!showNewFile)}
            className="p-1 rounded hover:bg-gray-100"
            title="New file"
          >
            <Plus className="h-4 w-4 text-gray-600" />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
          {error}
        </div>
      )}

      {showNewFile && (
        <div className="mb-3 p-2 bg-gray-50 rounded">
          <input
            type="text"
            placeholder="config-name.yaml"
            value={newFileName}
            onChange={(e) => setNewFileName(e.target.value)}
            className="w-full text-xs p-1 border border-gray-300 rounded"
            onKeyPress={(e) => e.key === 'Enter' && handleCreateFile()}
          />
          <div className="flex space-x-1 mt-1">
            <button
              onClick={handleCreateFile}
              className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Create
            </button>
            <button
              onClick={() => {
                setShowNewFile(false);
                setNewFileName('');
                setError(null);
              }}
              className="text-xs px-2 py-1 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* File List */}
      <div className="space-y-1 mb-3 max-h-32 overflow-y-auto">
        {configFiles.length === 0 && !loading && !error ? (
          <div className="text-center py-4">
            <FileText className="h-8 w-8 text-gray-300 mx-auto mb-2" />
            <p className="text-xs text-gray-500">No configuration files found</p>
            <p className="text-xs text-gray-400">Backend may not be running</p>
          </div>
        ) : (
          configFiles.map((file) => (
            <div
              key={file.filename}
              className={`flex items-center justify-between p-2 rounded cursor-pointer text-xs ${
                selectedFile === file.filename
                  ? 'bg-blue-50 border border-blue-200'
                  : 'hover:bg-gray-50'
              }`}
              onClick={() => handleFileSelect(file.filename)}
            >
              <div className="flex items-center space-x-2 flex-1 min-w-0">
                <FileText className="h-3 w-3 text-gray-500 flex-shrink-0" />
                {file.isPinned && <Pin className="h-3 w-3 text-blue-500 flex-shrink-0" />}
                <span className="truncate">{file.filename}</span>
                {file.modified && (
                  <div className="h-2 w-2 bg-orange-500 rounded-full flex-shrink-0" title="Modified" />
                )}
              </div>
              
              <div className="flex space-x-1 ml-2">
                {file.modified && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSaveFile(file.filename);
                    }}
                    className="p-1 rounded hover:bg-blue-100"
                    title="Save"
                  >
                    <Save className="h-3 w-3 text-blue-600" />
                  </button>
                )}
                
                {!file.isPinned && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteFile(file.filename);
                    }}
                    className="p-1 rounded hover:bg-red-100"
                    title="Delete"
                  >
                    <Trash2 className="h-3 w-3 text-red-600" />
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Editor */}
      {selectedFileData && (
        <div className="border border-gray-200 rounded">
          <div className="flex items-center justify-between p-2 bg-gray-50 border-b border-gray-200">
            <span className="text-xs font-medium text-gray-700">{selectedFile}</span>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setEditMode(editMode === 'visual' ? 'raw' : 'visual')}
                className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
              >
                {editMode === 'visual' ? 'Raw' : 'Visual'}
              </button>
              <Edit className="h-3 w-3 text-gray-500" />
            </div>
          </div>
          
          {editMode === 'visual' ? (
            <div className="p-2 max-h-48 overflow-y-auto">
              <div className="space-y-2">
                {selectedFileData.params.map((param, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={param.key}
                      onChange={(e) => {
                        const updatedParams = [...selectedFileData.params];
                        updatedParams[index] = { ...param, key: e.target.value };
                        setConfigFiles(prev => 
                          prev.map(file => 
                            file.filename === selectedFile 
                              ? { 
                                  ...file, 
                                  params: updatedParams,
                                  content: paramsToYaml(updatedParams, selectedFile),
                                  modified: true 
                                }
                              : file
                          )
                        );
                      }}
                      className="w-24 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                      placeholder="Parameter name"
                    />
                    
                    {renderParamInput(param, selectedFile)}
                    
                    <button
                      onClick={() => handleRemoveParam(selectedFile, param.key)}
                      className="p-1 rounded hover:bg-red-100"
                      title="Remove parameter"
                    >
                      <Trash2 className="h-3 w-3 text-red-600" />
                    </button>
                  </div>
                ))}
                
                <button
                  onClick={() => handleAddParam(selectedFile)}
                  className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200 flex items-center space-x-1"
                >
                  <Plus className="h-3 w-3" />
                  <span>Add Parameter</span>
                </button>
              </div>
            </div>
          ) : (
            <textarea
              value={selectedFileData.content}
              onChange={(e) => handleRawContentChange(selectedFile, e.target.value)}
              className="w-full h-48 p-2 text-xs font-mono border-0 resize-none focus:outline-none"
              placeholder="Edit YAML content..."
            />
          )}
          
          <div className="flex justify-between items-center p-2 bg-gray-50 border-t border-gray-200">
            <div className="flex items-center space-x-1">
              {selectedFileData.modified ? (
                <AlertCircle className="h-3 w-3 text-orange-500" />
              ) : (
                <CheckCircle className="h-3 w-3 text-green-500" />
              )}
              <span className="text-xs text-gray-600">
                {selectedFileData.modified ? 'Modified' : 'Saved'}
              </span>
            </div>
            
            {selectedFileData.modified && (
              <button
                onClick={() => handleSaveFile(selectedFile)}
                className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Save
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ConfigEditor;