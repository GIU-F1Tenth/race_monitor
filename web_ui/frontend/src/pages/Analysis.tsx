import React, { useState, useEffect } from 'react';
import { GitCompare, Play, BarChart3, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react';

interface ExperimentData {
  id: string;
  name: string;
  date: string;
  controller?: string;
  status: string;
}

interface EvoMetrics {
  evo_available: boolean;
  metrics?: {
    ape_mean?: number;
    ape_std?: number;
    ape_max?: number;
    rpe_mean?: number;
    rpe_std?: number;
    rpe_max?: number;
  };
  analysis_timestamp?: string;
  trajectory_length?: number;
}

interface AnalysisResult {
  status: 'pending' | 'running' | 'completed' | 'error';
  message?: string;
  results?: EvoMetrics;
  timestamp?: string;
}

const Analysis: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentData[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<string>('');
  const [compareExperiment, setCompareExperiment] = useState<string>('');
  const [evoMetrics, setEvoMetrics] = useState<EvoMetrics | null>(null);
  const [comparisonResult, setComparisonResult] = useState<any>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchExperiments();
  }, []);

  useEffect(() => {
    if (selectedExperiment) {
      fetchEvoMetrics(selectedExperiment);
    }
  }, [selectedExperiment]);

  const fetchExperiments = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/data/experiments');
      const data = await response.json();
      setExperiments(data.experiments || []);
      
      // Select first experiment by default
      if (data.experiments && data.experiments.length > 0) {
        setSelectedExperiment(data.experiments[0].id);
      }
    } catch (err) {
      setError('Failed to fetch experiments');
      console.error('Error fetching experiments:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchEvoMetrics = async (experimentId: string) => {
    try {
      const response = await fetch(`/api/evo/metrics/${experimentId}`);
      const data = await response.json();
      setEvoMetrics(data);
    } catch (err) {
      console.error('Error fetching EVO metrics:', err);
      setEvoMetrics({ evo_available: false });
    }
  };

  const runEvoAnalysis = async (experimentId: string) => {
    try {
      setAnalyzing(true);
      setAnalysisResult({ status: 'running', message: 'Running EVO analysis...' });
      
      const response = await fetch(`/api/evo/analyze/${experimentId}`, {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setAnalysisResult({
          status: 'completed',
          message: 'Analysis completed successfully',
          results: data,
          timestamp: new Date().toISOString()
        });
        // Refresh metrics
        await fetchEvoMetrics(experimentId);
      } else {
        setAnalysisResult({
          status: 'error',
          message: data.detail || 'Analysis failed'
        });
      }
    } catch (err) {
      setAnalysisResult({
        status: 'error',
        message: 'Network error during analysis'
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const compareExperiments = async () => {
    if (!selectedExperiment || !compareExperiment) return;
    
    try {
      setComparing(true);
      const response = await fetch(
        `/api/evo/compare?exp1=${selectedExperiment}&exp2=${compareExperiment}`
      );
      const data = await response.json();
      
      if (response.ok) {
        setComparisonResult(data);
      } else {
        setError(data.detail || 'Comparison failed');
      }
    } catch (err) {
      setError('Network error during comparison');
    } finally {
      setComparing(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const formatMetric = (value: number | undefined, suffix: string = '') => {
    if (value === undefined) return 'N/A';
    return `${value.toFixed(6)}${suffix}`;
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <p className="text-red-600">{error}</p>
        <button 
          onClick={() => {
            setError('');
            fetchExperiments();
          }}
          className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  const selectedExp = experiments.find(exp => exp.id === selectedExperiment);
  const compareExp = experiments.find(exp => exp.id === compareExperiment);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">EVO Analysis & Comparison</h1>
        <div className="flex space-x-2">
          {selectedExperiment && (
            <button
              onClick={() => runEvoAnalysis(selectedExperiment)}
              disabled={analyzing}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center space-x-2"
            >
              {analyzing ? (
                <RefreshCw className="h-5 w-5 animate-spin" />
              ) : (
                <Play className="h-5 w-5" />
              )}
              <span>{analyzing ? 'Analyzing...' : 'Run Analysis'}</span>
            </button>
          )}
          {selectedExperiment && compareExperiment && (
            <button
              onClick={compareExperiments}
              disabled={comparing}
              className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:bg-gray-400 flex items-center space-x-2"
            >
              {comparing ? (
                <RefreshCw className="h-5 w-5 animate-spin" />
              ) : (
                <GitCompare className="h-5 w-5" />
              )}
              <span>{comparing ? 'Comparing...' : 'Compare'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Analysis Status */}
      {analysisResult && (
        <div className={`p-4 rounded-lg border ${
          analysisResult.status === 'completed' 
            ? 'bg-green-50 border-green-200' 
            : analysisResult.status === 'error'
            ? 'bg-red-50 border-red-200'
            : 'bg-blue-50 border-blue-200'
        }`}>
          <div className="flex items-center space-x-2">
            {analysisResult.status === 'completed' && <CheckCircle className="h-5 w-5 text-green-600" />}
            {analysisResult.status === 'error' && <AlertCircle className="h-5 w-5 text-red-600" />}
            {analysisResult.status === 'running' && <RefreshCw className="h-5 w-5 text-blue-600 animate-spin" />}
            <span className={`font-medium ${
              analysisResult.status === 'completed' 
                ? 'text-green-800' 
                : analysisResult.status === 'error'
                ? 'text-red-800'
                : 'text-blue-800'
            }`}>
              {analysisResult.message}
            </span>
          </div>
          {analysisResult.timestamp && (
            <p className="text-sm text-gray-600 mt-1">
              {new Date(analysisResult.timestamp).toLocaleString()}
            </p>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Experiment Selection */}
        <div className="space-y-6">
          {/* Primary Experiment */}
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Primary Experiment</h2>
            </div>
            <div className="p-6">
              <select
                value={selectedExperiment}
                onChange={(e) => setSelectedExperiment(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Select an experiment</option>
                {experiments.map((exp) => (
                  <option key={exp.id} value={exp.id}>
                    {exp.name} - {formatDate(exp.date)}
                  </option>
                ))}
              </select>
              
              {selectedExp && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-900">{selectedExp.name}</h3>
                  <p className="text-sm text-gray-600">Date: {formatDate(selectedExp.date)}</p>
                  {selectedExp.controller && (
                    <p className="text-sm text-gray-600">Controller: {selectedExp.controller}</p>
                  )}
                  <span className={`inline-block mt-2 px-2 py-1 text-xs rounded-full ${
                    selectedExp.status === 'completed' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {selectedExp.status}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Comparison Experiment */}
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Compare With</h2>
            </div>
            <div className="p-6">
              <select
                value={compareExperiment}
                onChange={(e) => setCompareExperiment(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Select experiment to compare</option>
                {experiments
                  .filter(exp => exp.id !== selectedExperiment)
                  .map((exp) => (
                    <option key={exp.id} value={exp.id}>
                      {exp.name} - {formatDate(exp.date)}
                    </option>
                  ))}
              </select>
              
              {compareExp && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-900">{compareExp.name}</h3>
                  <p className="text-sm text-gray-600">Date: {formatDate(compareExp.date)}</p>
                  {compareExp.controller && (
                    <p className="text-sm text-gray-600">Controller: {compareExp.controller}</p>
                  )}
                  <span className={`inline-block mt-2 px-2 py-1 text-xs rounded-full ${
                    compareExp.status === 'completed' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {compareExp.status}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* EVO Metrics */}
        <div className="space-y-6">
          {evoMetrics && (
            <div className="bg-white rounded-lg shadow-md border border-gray-200">
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold text-gray-900">EVO Metrics</h2>
                  <div className={`flex items-center space-x-2 ${
                    evoMetrics.evo_available ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {evoMetrics.evo_available ? (
                      <CheckCircle className="h-5 w-5" />
                    ) : (
                      <AlertCircle className="h-5 w-5" />
                    )}
                    <span className="text-sm font-medium">
                      {evoMetrics.evo_available ? 'Available' : 'Not Available'}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                {evoMetrics.evo_available && evoMetrics.metrics ? (
                  <div className="space-y-4">
                    {/* APE Metrics */}
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3">
                        Absolute Pose Error (APE)
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-blue-50 p-3 rounded-lg">
                          <p className="text-sm font-medium text-blue-600">Mean</p>
                          <p className="text-lg font-bold text-blue-900">
                            {formatMetric(evoMetrics.metrics.ape_mean, 'm')}
                          </p>
                        </div>
                        <div className="bg-yellow-50 p-3 rounded-lg">
                          <p className="text-sm font-medium text-yellow-600">Std Dev</p>
                          <p className="text-lg font-bold text-yellow-900">
                            {formatMetric(evoMetrics.metrics.ape_std, 'm')}
                          </p>
                        </div>
                        <div className="bg-red-50 p-3 rounded-lg">
                          <p className="text-sm font-medium text-red-600">Max</p>
                          <p className="text-lg font-bold text-red-900">
                            {formatMetric(evoMetrics.metrics.ape_max, 'm')}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* RPE Metrics */}
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-3">
                        Relative Pose Error (RPE)
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="bg-green-50 p-3 rounded-lg">
                          <p className="text-sm font-medium text-green-600">Mean</p>
                          <p className="text-lg font-bold text-green-900">
                            {formatMetric(evoMetrics.metrics.rpe_mean, 'm')}
                          </p>
                        </div>
                        <div className="bg-purple-50 p-3 rounded-lg">
                          <p className="text-sm font-medium text-purple-600">Std Dev</p>
                          <p className="text-lg font-bold text-purple-900">
                            {formatMetric(evoMetrics.metrics.rpe_std, 'm')}
                          </p>
                        </div>
                        <div className="bg-orange-50 p-3 rounded-lg">
                          <p className="text-sm font-medium text-orange-600">Max</p>
                          <p className="text-lg font-bold text-orange-900">
                            {formatMetric(evoMetrics.metrics.rpe_max, 'm')}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Additional Info */}
                    {(evoMetrics.trajectory_length || evoMetrics.analysis_timestamp) && (
                      <div className="pt-4 border-t border-gray-200">
                        {evoMetrics.trajectory_length && (
                          <p className="text-sm text-gray-600">
                            Trajectory Length: {evoMetrics.trajectory_length.toFixed(2)}m
                          </p>
                        )}
                        {evoMetrics.analysis_timestamp && (
                          <p className="text-sm text-gray-600">
                            Last Analysis: {new Date(evoMetrics.analysis_timestamp).toLocaleString()}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">
                      {evoMetrics.evo_available 
                        ? 'No metrics available. Run analysis to generate EVO metrics.'
                        : 'EVO integration not available or experiment data not found.'
                      }
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Comparison Results */}
          {comparisonResult && (
            <div className="bg-white rounded-lg shadow-md border border-gray-200">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-900">Comparison Results</h2>
              </div>
              <div className="p-6">
                {comparisonResult.error ? (
                  <div className="text-center py-8">
                    <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
                    <p className="text-red-600">{comparisonResult.error}</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-gray-600 mb-4">
                      Comparison between {selectedExp?.name} and {compareExp?.name}
                    </p>
                    <div className="text-center py-8">
                      <GitCompare className="h-12 w-12 text-blue-400 mx-auto mb-4" />
                      <p className="text-gray-500">
                        Detailed comparison visualization will be displayed here
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Empty State */}
      {experiments.length === 0 && (
        <div className="bg-white rounded-lg shadow-md p-12 border border-gray-200 text-center">
          <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No Experiments Found</h2>
          <p className="text-gray-600">
            No experiment data available for analysis. Please ensure race data has been collected.
          </p>
        </div>
      )}
    </div>
  );
};

export default Analysis;