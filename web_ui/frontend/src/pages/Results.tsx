import React, { useState, useEffect } from 'react';
import { BarChart3, Clock, Trophy, Download } from 'lucide-react';

interface ExperimentData {
  id: string;
  name: string;
  date: string;
  laps?: number;
  status: string;
  controller?: string;
  metrics?: {
    avg_consistency?: number;
    avg_path_length?: number;
    avg_smoothness?: number;
  };
}

interface LapData {
  lap_number: number;
  lap_time_s: number;
  consistency?: number;
  smoothness?: number;
  path_length?: number;
}

interface SummaryData {
  experiments_count: number;
  total_laps: number;
  best_lap_time?: number;
  avg_lap_time?: number;
}

const Results: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentData[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<string>('');
  const [lapData, setLapData] = useState<LapData[]>([]);
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (selectedExperiment) {
      fetchExperimentDetails(selectedExperiment);
    }
  }, [selectedExperiment]);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Add timeout to prevent infinite loading
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      // Fetch experiments list
      const experimentsResponse = await fetch('/api/data/experiments', { signal: controller.signal });
      
      if (experimentsResponse.ok) {
        const experimentsData = await experimentsResponse.json();
        setExperiments(experimentsData.experiments || []);
        
        // Select first experiment by default
        if (experimentsData.experiments && experimentsData.experiments.length > 0) {
          setSelectedExperiment(experimentsData.experiments[0].id);
        }
      } else {
        // If no experiments found, set empty state
        setExperiments([]);
        setError('No race data found. Directory might be empty or backend not accessible.');
      }
      
      // Fetch summary
      const summaryResponse = await fetch('/api/data/summary', { signal: controller.signal });
      
      if (summaryResponse.ok) {
        const summaryData = await summaryResponse.json();
        setSummary(summaryData);
      } else {
        setSummary({
          experiments_count: 0,
          total_laps: 0
        });
      }
      
      clearTimeout(timeoutId);
      
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Please check your connection and try again.');
      } else {
        setError('Failed to connect to backend. Make sure the backend server is running.');
      }
      setExperiments([]);
      setSummary({
        experiments_count: 0,
        total_laps: 0
      });
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchExperimentDetails = async (experimentId: string) => {
    try {
      const response = await fetch(`/api/data/lap-analysis/${experimentId}`);
      if (response.ok) {
        const data = await response.json();
        setLapData(data.laps || []);
      }
    } catch (err) {
      console.error('Error fetching experiment details:', err);
    }
  };

  const exportExperiment = async (experimentId: string, format: string) => {
    try {
      const response = await fetch(`/api/files/export/${experimentId}?format=${format}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${experimentId}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      console.error('Error exporting experiment:', err);
    }
  };

  const formatTime = (seconds: number) => {
    return `${seconds.toFixed(3)}s`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getBestLapTime = (laps: LapData[]) => {
    if (laps.length === 0) return null;
    return Math.min(...laps.map(lap => lap.lap_time_s));
  };

  const getAverageLapTime = (laps: LapData[]) => {
    if (laps.length === 0) return null;
    return laps.reduce((sum, lap) => sum + lap.lap_time_s, 0) / laps.length;
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
          onClick={fetchData}
          className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  // Show empty state when no experiments are found
  if (!loading && experiments.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">Race Results</h1>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-12 text-center">
          <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Race Results Found</h3>
          <p className="text-gray-600 mb-4">
            No experiments or race data available yet. Start a race session to see results here.
          </p>
          <button 
            onClick={fetchData}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Refresh
          </button>
        </div>
      </div>
    );
  }

  const selectedExp = experiments.find(exp => exp.id === selectedExperiment);
  const bestLap = getBestLapTime(lapData);
  const avgLap = getAverageLapTime(lapData);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Race Results</h1>
        <div className="flex space-x-2">
          {selectedExperiment && (
            <>
              <button
                onClick={() => exportExperiment(selectedExperiment, 'csv')}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center space-x-2"
              >
                <Download className="h-5 w-5" />
                <span>Export CSV</span>
              </button>
              <button
                onClick={() => exportExperiment(selectedExperiment, 'json')}
                className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 flex items-center space-x-2"
              >
                <Download className="h-5 w-5" />
                <span>Export JSON</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
            <div className="flex items-center">
              <BarChart3 className="h-8 w-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Experiments</p>
                <p className="text-2xl font-bold text-gray-900">{summary.experiments_count}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
            <div className="flex items-center">
              <Trophy className="h-8 w-8 text-yellow-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Laps</p>
                <p className="text-2xl font-bold text-gray-900">{summary.total_laps}</p>
              </div>
            </div>
          </div>
          
          {bestLap && (
            <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Best Lap Time</p>
                  <p className="text-2xl font-bold text-gray-900">{formatTime(bestLap)}</p>
                </div>
              </div>
            </div>
          )}
          
          {avgLap && (
            <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-blue-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Average Lap Time</p>
                  <p className="text-2xl font-bold text-gray-900">{formatTime(avgLap)}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Experiments List */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Experiments</h2>
            </div>
            <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
              {experiments.map((experiment) => (
                <div
                  key={experiment.id}
                  onClick={() => setSelectedExperiment(experiment.id)}
                  className={`p-4 rounded-lg cursor-pointer transition-colors ${
                    selectedExperiment === experiment.id
                      ? 'bg-blue-50 border-2 border-blue-200'
                      : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium text-gray-900">{experiment.name}</h3>
                      <p className="text-sm text-gray-500">{formatDate(experiment.date)}</p>
                      {experiment.controller && (
                        <p className="text-sm text-blue-600">{experiment.controller}</p>
                      )}
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      experiment.status === 'completed' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {experiment.status}
                    </span>
                  </div>
                  {experiment.laps && (
                    <p className="text-sm text-gray-600 mt-1">{experiment.laps} laps</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Experiment Details */}
        <div className="lg:col-span-2">
          {selectedExp ? (
            <div className="space-y-6">
              {/* Experiment Info */}
              <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">{selectedExp.name}</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Date</p>
                    <p className="text-gray-900">{formatDate(selectedExp.date)}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-500">Status</p>
                    <p className="text-gray-900">{selectedExp.status}</p>
                  </div>
                  {selectedExp.controller && (
                    <div>
                      <p className="text-sm font-medium text-gray-500">Controller</p>
                      <p className="text-gray-900">{selectedExp.controller}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm font-medium text-gray-500">Total Laps</p>
                    <p className="text-gray-900">{lapData.length}</p>
                  </div>
                </div>

                {selectedExp.metrics && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h3 className="text-lg font-medium text-gray-900 mb-3">Performance Metrics</h3>
                    <div className="grid grid-cols-3 gap-4">
                      {selectedExp.metrics.avg_consistency !== undefined && (
                        <div>
                          <p className="text-sm font-medium text-gray-500">Consistency</p>
                          <p className="text-gray-900">{selectedExp.metrics.avg_consistency.toFixed(3)}</p>
                        </div>
                      )}
                      {selectedExp.metrics.avg_path_length !== undefined && (
                        <div>
                          <p className="text-sm font-medium text-gray-500">Path Length</p>
                          <p className="text-gray-900">{selectedExp.metrics.avg_path_length.toFixed(2)}m</p>
                        </div>
                      )}
                      {selectedExp.metrics.avg_smoothness !== undefined && (
                        <div>
                          <p className="text-sm font-medium text-gray-500">Smoothness</p>
                          <p className="text-gray-900">{selectedExp.metrics.avg_smoothness.toFixed(3)}</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Lap Times Table */}
              {lapData.length > 0 && (
                <div className="bg-white rounded-lg shadow-md border border-gray-200">
                  <div className="p-6 border-b border-gray-200">
                    <h3 className="text-xl font-semibold text-gray-900">Lap Times</h3>
                  </div>
                  <div className="p-6">
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Lap
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Time
                            </th>
                            {lapData[0]?.consistency !== undefined && (
                              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Consistency
                              </th>
                            )}
                            {lapData[0]?.smoothness !== undefined && (
                              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Smoothness
                              </th>
                            )}
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Relative
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {lapData.map((lap) => {
                            const isBestLap = lap.lap_time_s === bestLap;
                            const timeDiff = bestLap ? lap.lap_time_s - bestLap : 0;
                            
                            return (
                              <tr key={lap.lap_number} className={isBestLap ? 'bg-green-50' : ''}>
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                  #{lap.lap_number}
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                  <span className={isBestLap ? 'font-bold text-green-600' : ''}>
                                    {formatTime(lap.lap_time_s)}
                                  </span>
                                </td>
                                {lap.consistency !== undefined && (
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    {lap.consistency.toFixed(3)}
                                  </td>
                                )}
                                {lap.smoothness !== undefined && (
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    {lap.smoothness.toFixed(3)}
                                  </td>
                                )}
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                  {timeDiff === 0 ? (
                                    <span className="text-green-600 font-medium">Best</span>
                                  ) : (
                                    <span className={timeDiff > 1 ? 'text-red-600' : 'text-yellow-600'}>
                                      +{formatTime(timeDiff)}
                                    </span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
              <p className="text-gray-500 text-center">Select an experiment to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Results;