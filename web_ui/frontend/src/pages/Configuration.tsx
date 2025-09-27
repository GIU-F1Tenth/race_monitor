import React, { useState, useEffect } from 'react';
import { 
  Save, 
  Upload, 
  Download, 
  Plus, 
  X, 
  FileText, 
  AlertCircle,
  CheckCircle,
  RefreshCw
} from 'lucide-react';
import Editor from '@monaco-editor/react';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import * as yaml from 'js-yaml';

interface ConfigFile {
  filename: string;
  size: number;
  modified: string;
  is_template: boolean;
}

interface ConfigData {
  configs: ConfigFile[];
  templates: Record<string, string>;
  config_dir: string;
}

const Configuration: React.FC = () => {
  const [configs, setConfigs] = useState<ConfigFile[]>([]);
  const [activeTab, setActiveTab] = useState<string>('');
  const [editorContent, setEditorContent] = useState<string>('');
  const [originalContent, setOriginalContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [yamlValid, setYamlValid] = useState(true);
  const [yamlError, setYamlError] = useState<string>('');

  // Fetch available configurations
  const fetchConfigs = async () => {
    try {
      const response = await axios.get<ConfigData>('/api/config/list');
      setConfigs(response.data.configs);
      
      // Set default tab to race_monitor.yaml if available
      if (response.data.configs.length > 0 && !activeTab) {
        const defaultConfig = response.data.configs.find(c => c.filename === 'race_monitor.yaml') 
                           || response.data.configs[0];
        setActiveTab(defaultConfig.filename);
        await loadConfig(defaultConfig.filename);
      }
    } catch (error) {
      toast.error('Failed to fetch configurations');
      console.error('Error fetching configs:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load specific configuration file
  const loadConfig = async (filename: string) => {
    try {
      const response = await axios.get(`/api/config/${filename}`);
      const content = response.data.content;
      setEditorContent(content);
      setOriginalContent(content);
      setHasChanges(false);
      validateYaml(content);
    } catch (error) {
      toast.error(`Failed to load ${filename}`);
      console.error('Error loading config:', error);
    }
  };

  // Validate YAML content
  const validateYaml = (content: string) => {
    try {
      yaml.load(content);
      setYamlValid(true);
      setYamlError('');
    } catch (error) {
      setYamlValid(false);
      setYamlError(error instanceof Error ? error.message : 'Invalid YAML');
    }
  };

  // Handle editor content change
  const handleEditorChange = (value: string | undefined) => {
    const newContent = value || '';
    setEditorContent(newContent);
    setHasChanges(newContent !== originalContent);
    validateYaml(newContent);
  };

  // Save current configuration
  const saveConfig = async () => {
    if (!activeTab || !yamlValid) return;

    setSaving(true);
    try {
      await axios.post(`/api/config/${activeTab}`, {
        content: editorContent,
        filename: activeTab
      });
      
      setOriginalContent(editorContent);
      setHasChanges(false);
      toast.success(`Saved ${activeTab}`);
      
      // Refresh config list
      await fetchConfigs();
    } catch (error) {
      toast.error('Failed to save configuration');
      console.error('Error saving config:', error);
    } finally {
      setSaving(false);
    }
  };

  // Handle tab change
  const handleTabChange = async (filename: string) => {
    if (hasChanges) {
      const confirm = window.confirm('You have unsaved changes. Are you sure you want to switch tabs?');
      if (!confirm) return;
    }
    
    setActiveTab(filename);
    await loadConfig(filename);
  };

  // Upload new configuration file
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('/api/config/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      toast.success(`Uploaded ${file.name}`);
      await fetchConfigs();
      
      // Switch to uploaded file
      setActiveTab(file.name);
      await loadConfig(file.name);
    } catch (error) {
      toast.error('Failed to upload file');
      console.error('Error uploading file:', error);
    }
  };

  // Download current configuration
  const downloadConfig = () => {
    if (!activeTab || !editorContent) return;
    
    const blob = new Blob([editorContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = activeTab;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-race-primary" />
        <span className="ml-2 text-gray-600">Loading configurations...</span>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Configuration Editor</h1>
        <div className="flex items-center space-x-2">
          {/* Upload Button */}
          <label className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors cursor-pointer">
            <Upload className="h-4 w-4" />
            <span>Upload</span>
            <input
              type="file"
              accept=".yaml,.yml"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          {/* Download Button */}
          <button
            onClick={downloadConfig}
            disabled={!activeTab}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Download className="h-4 w-4" />
            <span>Download</span>
          </button>

          {/* Save Button */}
          <button
            onClick={saveConfig}
            disabled={!hasChanges || !yamlValid || saving}
            className="flex items-center space-x-2 px-4 py-2 bg-race-primary text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {saving ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            <span>{saving ? 'Saving...' : 'Save'}</span>
          </button>
        </div>
      </div>

      <div className="flex-1 flex bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
        {/* Config File Tabs (Right Sidebar) */}
        <div className="w-80 border-r border-gray-200 bg-gray-50">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Configuration Files</h2>
          </div>
          
          <div className="overflow-y-auto h-full">
            {configs.map((config) => (
              <button
                key={config.filename}
                onClick={() => handleTabChange(config.filename)}
                className={`
                  w-full text-left p-4 border-b border-gray-100 hover:bg-gray-100 transition-colors
                  ${activeTab === config.filename ? 'bg-race-primary/10 border-r-4 border-r-race-primary' : ''}
                `}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <FileText className="h-4 w-4 flex-shrink-0 text-gray-500" />
                      <span className={`text-sm font-medium truncate ${
                        activeTab === config.filename ? 'text-race-primary' : 'text-gray-900'
                      }`}>
                        {config.filename}
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-gray-500">
                      <div>Size: {(config.size / 1024).toFixed(1)} KB</div>
                      <div>Modified: {new Date(config.modified).toLocaleDateString()}</div>
                    </div>
                    {config.is_template && (
                      <div className="mt-1">
                        <span className="inline-flex px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                          Template
                        </span>
                      </div>
                    )}
                  </div>
                  
                  {activeTab === config.filename && hasChanges && (
                    <div className="flex-shrink-0 ml-2">
                      <div className="h-2 w-2 bg-orange-500 rounded-full"></div>
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Editor Area */}
        <div className="flex-1 flex flex-col">
          {/* Editor Header */}
          {activeTab && (
            <div className="p-4 border-b border-gray-200 bg-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <h3 className="text-lg font-semibold text-gray-900">{activeTab}</h3>
                  {hasChanges && (
                    <span className="text-sm text-orange-600 font-medium">• Unsaved changes</span>
                  )}
                </div>
                
                <div className="flex items-center space-x-2">
                  {yamlValid ? (
                    <div className="flex items-center space-x-1 text-green-600">
                      <CheckCircle className="h-4 w-4" />
                      <span className="text-sm">Valid YAML</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-1 text-red-600">
                      <AlertCircle className="h-4 w-4" />
                      <span className="text-sm">Invalid YAML</span>
                    </div>
                  )}
                </div>
              </div>
              
              {!yamlValid && yamlError && (
                <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-sm text-red-700">{yamlError}</p>
                </div>
              )}
            </div>
          )}

          {/* Monaco Editor */}
          <div className="flex-1">
            {activeTab ? (
              <Editor
                height="100%"
                language="yaml"
                value={editorContent}
                onChange={handleEditorChange}
                theme="vs-light"
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  lineNumbers: 'on',
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  wordWrap: 'on',
                  folding: true,
                  renderWhitespace: 'selection',
                  tabSize: 2,
                  insertSpaces: true
                }}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <FileText className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-lg font-medium">No configuration file selected</p>
                  <p className="text-sm">Select a file from the sidebar to start editing</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Configuration;