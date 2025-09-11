'use client';

import React from 'react';
import { MetricsComparisonProps } from './types';
import { formatMetricValue, isHigherBetter, calculateRelativeImprovement } from './utils';

export const MetricsComparison: React.FC<MetricsComparisonProps> = ({
  evaluations,
  selectedMetrics,
  onMetricToggle,
  comparisonMode,
  onComparisonModeChange,
  className = '',
}) => {
  const availableMetrics = [
    { key: 'map', name: 'mAP', category: 'Detection' },
    { key: 'map_50', name: 'mAP@50', category: 'Detection' },
    { key: 'map_75', name: 'mAP@75', category: 'Detection' },
    { key: 'accuracy', name: 'Accuracy', category: 'Classification' },
    { key: 'precision', name: 'Precision', category: 'General' },
    { key: 'recall', name: 'Recall', category: 'General' },
    { key: 'f1_score', name: 'F1 Score', category: 'General' },
    { key: 'inference_time_mean', name: 'Inference Time', category: 'Performance' },
    { key: 'fps', name: 'FPS', category: 'Performance' },
    { key: 'memory_usage_peak', name: 'Peak Memory', category: 'Performance' },
    { key: 'model_size', name: 'Model Size', category: 'Performance' },
  ];

  const completedEvaluations = evaluations.filter(e => e.status === 'completed');

  const getMetricValue = (evaluation: any, metricKey: string) => {
    return evaluation.metrics[metricKey];
  };

  const getBestValue = (metricKey: string) => {
    const values = completedEvaluations
      .map(e => getMetricValue(e, metricKey))
      .filter(v => v !== undefined && v !== null);
    
    if (values.length === 0) return null;
    
    return isHigherBetter(metricKey) ? Math.max(...values) : Math.min(...values);
  };

  const getBestEvaluation = (metricKey: string) => {
    const bestValue = getBestValue(metricKey);
    if (bestValue === null) return null;
    
    return completedEvaluations.find(e => getMetricValue(e, metricKey) === bestValue);
  };

  if (completedEvaluations.length === 0) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center ${className}`}>
        <div className="w-16 h-16 mx-auto mb-4 text-gray-400">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          No Completed Evaluations
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Complete some model evaluations to see performance comparisons.
        </p>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Metrics Comparison ({completedEvaluations.length} models)
          </h2>
          
          {/* Comparison Mode Toggle */}
          <div className="flex rounded-md border border-gray-300 dark:border-gray-600">
            {(['table', 'chart', 'radar'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => onComparisonModeChange(mode)}
                className={`px-3 py-1 text-sm first:rounded-l-md last:rounded-r-md border-r border-gray-300 dark:border-gray-600 last:border-r-0 transition-colors ${
                  comparisonMode === mode
                    ? 'bg-blue-500 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-750'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Metric Selection */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Select Metrics to Compare:
          </h4>
          <div className="flex flex-wrap gap-2">
            {availableMetrics.map((metric) => (
              <button
                key={metric.key}
                onClick={() => onMetricToggle(metric.key)}
                className={`px-3 py-1 text-xs rounded-full transition-colors ${
                  selectedMetrics.includes(metric.key)
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {metric.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Comparison Content */}
      <div className="p-6">
        {comparisonMode === 'table' && (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-900 dark:text-gray-100">
                    Model
                  </th>
                  {selectedMetrics.map((metricKey) => {
                    const metric = availableMetrics.find(m => m.key === metricKey);
                    return (
                      <th key={metricKey} className="text-center py-3 px-4 font-medium text-gray-900 dark:text-gray-100">
                        {metric?.name}
                        <div className="text-xs font-normal text-gray-500 dark:text-gray-400">
                          {isHigherBetter(metricKey) ? '↑ Higher' : '↓ Lower'} is better
                        </div>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {completedEvaluations.map((evaluation, index) => (
                  <tr key={evaluation.id} className={`border-b border-gray-100 dark:border-gray-800 ${index % 2 === 0 ? 'bg-gray-50 dark:bg-gray-750' : ''}`}>
                    <td className="py-3 px-4">
                      <div>
                        <div className="font-medium text-gray-900 dark:text-gray-100">
                          {evaluation.model_name}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          {evaluation.name}
                        </div>
                      </div>
                    </td>
                    {selectedMetrics.map((metricKey) => {
                      const value = getMetricValue(evaluation, metricKey);
                      const bestValue = getBestValue(metricKey);
                      const isBest = value === bestValue && bestValue !== null;
                      
                      return (
                        <td key={metricKey} className="py-3 px-4 text-center">
                          <div className={`font-medium ${isBest ? 'text-green-600 dark:text-green-400' : 'text-gray-900 dark:text-gray-100'}`}>
                            {value !== undefined && value !== null 
                              ? formatMetricValue(value, metricKey)
                              : 'N/A'
                            }
                          </div>
                          {isBest && (
                            <div className="text-xs text-green-600 dark:text-green-400">
                              🏆 Best
                            </div>
                          )}
                          {value !== undefined && value !== null && bestValue !== null && !isBest && (
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              {(() => {
                                const improvement = calculateRelativeImprovement(value, bestValue, metricKey);
                                const sign = improvement >= 0 ? '+' : '';
                                return `${sign}${(improvement * 100).toFixed(1)}%`;
                              })()}
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {comparisonMode === 'chart' && (
          <div className="space-y-6">
            {selectedMetrics.map((metricKey) => {
              const metric = availableMetrics.find(m => m.key === metricKey);
              const values = completedEvaluations.map(e => ({
                name: e.model_name,
                value: getMetricValue(e, metricKey) || 0
              }));
              
              const maxValue = Math.max(...values.map(v => v.value));
              
              return (
                <div key={metricKey}>
                  <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
                    {metric?.name}
                  </h4>
                  <div className="space-y-2">
                    {values.map((item, index) => (
                      <div key={index} className="flex items-center">
                        <div className="w-32 text-sm text-gray-600 dark:text-gray-400 truncate mr-4">
                          {item.name}
                        </div>
                        <div className="flex-1 relative">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-6">
                            <div
                              className="bg-blue-500 h-6 rounded-full flex items-center justify-end pr-2"
                              style={{ width: `${maxValue > 0 ? (item.value / maxValue) * 100 : 0}%` }}
                            >
                              <span className="text-xs font-medium text-white">
                                {formatMetricValue(item.value, metricKey)}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {comparisonMode === 'radar' && (
          <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
            <div className="text-center">
              <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <p>Radar chart visualization would be implemented here</p>
              <p className="text-sm mt-2">Requires a charting library like Chart.js or D3</p>
            </div>
          </div>
        )}
      </div>

      {/* Summary */}
      <div className="p-6 bg-gray-50 dark:bg-gray-750 border-t border-gray-200 dark:border-gray-700">
        <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
          Performance Leaders
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {selectedMetrics.slice(0, 3).map((metricKey) => {
            const metric = availableMetrics.find(m => m.key === metricKey);
            const bestEval = getBestEvaluation(metricKey);
            const bestValue = bestEval ? getMetricValue(bestEval, metricKey) : null;
            
            return (
              <div key={metricKey} className="bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-700">
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Best {metric?.name}
                </div>
                {bestEval && bestValue !== null ? (
                  <>
                    <div className="text-lg font-bold text-green-600 dark:text-green-400">
                      {formatMetricValue(bestValue, metricKey)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {bestEval.model_name}
                    </div>
                  </>
                ) : (
                  <div className="text-sm text-gray-400">No data</div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};