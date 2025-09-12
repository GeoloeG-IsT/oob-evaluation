'use client';

import React, { useState, useMemo } from 'react';
import { EvaluationDashboardProps, EvaluationResult } from './types';
import { MetricsComparison } from './MetricsComparison';
import { EvaluationCard } from './EvaluationCard';
import { formatMetricValue, getMetricCategory, calculateRelativeImprovement } from './utils';

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  evaluations,
  selectedEvaluations = [],
  onEvaluationSelect,
  onEvaluationDelete,
  onEvaluationCreate,
  onEvaluationExport,
  className = '',
}) => {
  const [viewMode, setViewMode] = useState<'grid' | 'comparison' | 'trends'>('grid');
  const [filterBy, setFilterBy] = useState<'all' | 'detection' | 'segmentation' | 'classification'>('all');
  const [sortBy, setSortBy] = useState<'created_at' | 'map' | 'accuracy' | 'inference_time'>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['map', 'map_50', 'precision', 'recall']);
  const [comparisonMode, setComparisonMode] = useState<'table' | 'chart' | 'radar'>('table');

  // Filter and sort evaluations
  const processedEvaluations = useMemo(() => {
    let filtered = [...evaluations];

    // Apply type filter
    if (filterBy !== 'all') {
      filtered = filtered.filter(eval => eval.evaluation_type === filterBy);
    }

    // Sort evaluations
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'created_at':
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
        case 'map':
          comparison = (a.metrics.map || 0) - (b.metrics.map || 0);
          break;
        case 'accuracy':
          comparison = (a.metrics.accuracy || 0) - (b.metrics.accuracy || 0);
          break;
        case 'inference_time':
          comparison = (a.metrics.inference_time_mean || 0) - (b.metrics.inference_time_mean || 0);
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [evaluations, filterBy, sortBy, sortOrder]);

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    const completed = evaluations.filter(e => e.status === 'completed');
    const running = evaluations.filter(e => e.status === 'running');

    const avgMap = completed.reduce((sum, e) => sum + (e.metrics.map || 0), 0) / (completed.length || 1);
    const avgAccuracy = completed.reduce((sum, e) => sum + (e.metrics.accuracy || 0), 0) / (completed.length || 1);
    const avgInferenceTime = completed.reduce((sum, e) => sum + (e.metrics.inference_time_mean || 0), 0) / (completed.length || 1);

    return {
      total: evaluations.length,
      completed: completed.length,
      running: running.length,
      failed: evaluations.filter(e => e.status === 'failed').length,
      avgMap,
      avgAccuracy,
      avgInferenceTime,
    };
  }, [evaluations]);

  const handleMetricToggle = (metric: string) => {
    setSelectedMetrics(prev => 
      prev.includes(metric) 
        ? prev.filter(m => m !== metric)
        : [...prev, metric]
    );
  };

  const handleEvaluationSelect = (evaluationId: string) => {
    onEvaluationSelect?.(evaluationId);
  };

  const getBestPerforming = (metric: string) => {
    const completed = evaluations.filter(e => e.status === 'completed');
    if (completed.length === 0) return null;

    return completed.reduce((best, current) => {
      const currentValue = (current.metrics as any)[metric] || 0;
      const bestValue = (best.metrics as any)[metric] || 0;
      return currentValue > bestValue ? current : best;
    });
  };

  return (
    <div className={`evaluation-dashboard ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Model Evaluation Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Compare and analyze model performance metrics
          </p>
        </div>

        <div className="flex space-x-3">
          {onEvaluationCreate && (
            <button
              onClick={onEvaluationCreate}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
            >
              New Evaluation
            </button>
          )}
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 mb-8">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {summaryStats.total}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Total</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {summaryStats.completed}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Completed</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {summaryStats.running}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Running</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">
            {summaryStats.failed}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Failed</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {formatMetricValue(summaryStats.avgMap, 'map')}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Avg mAP</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
            {formatMetricValue(summaryStats.avgAccuracy, 'accuracy')}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Avg Accuracy</div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            {formatMetricValue(summaryStats.avgInferenceTime, 'inference_time')}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Avg Speed</div>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 mb-6">
        <div className="flex flex-wrap items-center gap-4">
          {/* View Mode */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">View:</label>
            <div className="flex rounded-md border border-gray-300 dark:border-gray-600">
              {(['grid', 'comparison', 'trends'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1 text-sm first:rounded-l-md last:rounded-r-md border-r border-gray-300 dark:border-gray-600 last:border-r-0 transition-colors ${
                    viewMode === mode
                      ? 'bg-blue-500 text-white'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-750'
                  }`}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Type Filter */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Type:</label>
            <select
              value={filterBy}
              onChange={(e) => setFilterBy(e.target.value as typeof filterBy)}
              className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="all">All Types</option>
              <option value="detection">Detection</option>
              <option value="segmentation">Segmentation</option>
              <option value="classification">Classification</option>
            </select>
          </div>

          {/* Sort Controls */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Sort:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="created_at">Date Created</option>
              <option value="map">mAP</option>
              <option value="accuracy">Accuracy</option>
              <option value="inference_time">Inference Time</option>
            </select>
            
            <button
              onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
              className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d={sortOrder === 'asc' 
                    ? "M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z"
                    : "M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                  }
                  clipRule="evenodd"
                />
              </svg>
            </button>
          </div>

          {/* Selected Count */}
          {selectedEvaluations.length > 0 && (
            <div className="flex items-center space-x-2">
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-sm">
                {selectedEvaluations.length} selected
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      {viewMode === 'grid' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {processedEvaluations.map((evaluation) => (
            <EvaluationCard
              key={evaluation.id}
              evaluation={evaluation}
              isSelected={selectedEvaluations.includes(evaluation.id)}
              onSelect={handleEvaluationSelect}
              onDelete={onEvaluationDelete}
              onExport={onEvaluationExport}
            />
          ))}
        </div>
      )}

      {viewMode === 'comparison' && (
        <MetricsComparison
          evaluations={selectedEvaluations.length > 0 
            ? evaluations.filter(e => selectedEvaluations.includes(e.id))
            : processedEvaluations.slice(0, 10)
          }
          selectedMetrics={selectedMetrics}
          onMetricToggle={handleMetricToggle}
          comparisonMode={comparisonMode}
          onComparisonModeChange={setComparisonMode}
        />
      )}

      {viewMode === 'trends' && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 text-gray-400">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              Trends Analysis
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Performance trends and historical analysis would be displayed here.
            </p>
          </div>
        </div>
      )}

      {/* Best Performers Sidebar */}
      {viewMode === 'grid' && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          {(['map', 'accuracy', 'inference_time_mean'] as const).map((metric) => {
            const best = getBestPerforming(metric);
            if (!best) return null;

            return (
              <div key={metric} className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900 dark:to-green-800 rounded-lg p-4 border border-green-200 dark:border-green-700">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
                    <span className="text-white font-bold text-sm">👑</span>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-green-800 dark:text-green-200">
                      Best {metric === 'map' ? 'mAP' : metric === 'accuracy' ? 'Accuracy' : 'Speed'}
                    </div>
                    <div className="text-xs text-green-600 dark:text-green-400">
                      {best.model_name}
                    </div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="text-2xl font-bold text-green-800 dark:text-green-200">
                    {formatMetricValue((best.metrics as any)[metric], metric)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {processedEvaluations.length === 0 && (
        <div className="text-center py-12">
          <div className="w-16 h-16 mx-auto mb-4 text-gray-400">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No Evaluations Found
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            {filterBy !== 'all' 
              ? `No ${filterBy} evaluations match your current filters.`
              : 'Create your first model evaluation to get started.'
            }
          </p>
          {onEvaluationCreate && (
            <button
              onClick={onEvaluationCreate}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
            >
              Create Evaluation
            </button>
          )}
        </div>
      )}
    </div>
  );
};