'use client';

import React, { useState } from 'react';
import { EvaluationResult } from './types';
import { formatMetricValue, formatTime } from './utils';

interface EvaluationCardProps {
  evaluation: EvaluationResult;
  isSelected?: boolean;
  onSelect?: (evaluationId: string) => void;
  onDelete?: (evaluationId: string) => void;
  onExport?: (evaluationId: string, format: string) => void;
  className?: string;
}

export const EvaluationCard: React.FC<EvaluationCardProps> = ({
  evaluation,
  isSelected = false,
  onSelect,
  onDelete,
  onExport,
  className = '',
}) => {
  const [showActions, setShowActions] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const getStatusColor = () => {
    switch (evaluation.status) {
      case 'running':
        return 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950';
      case 'completed':
        return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950';
      case 'failed':
        return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-950';
    }
  };

  const getStatusIcon = () => {
    switch (evaluation.status) {
      case 'running':
        return '🔄';
      case 'completed':
        return '✅';
      case 'failed':
        return '❌';
      default:
        return '⏳';
    }
  };

  const handleSelect = () => {
    onSelect?.(evaluation.id);
  };

  const handleDelete = () => {
    if (showDeleteConfirm) {
      onDelete?.(evaluation.id);
      setShowDeleteConfirm(false);
    } else {
      setShowDeleteConfirm(true);
    }
  };

  const handleExport = (format: string) => {
    onExport?.(evaluation.id, format);
  };

  const primaryMetrics = (() => {
    const { metrics } = evaluation;
    switch (evaluation.evaluation_type) {
      case 'detection':
        return [
          { name: 'mAP', value: metrics.map, key: 'map' },
          { name: 'mAP@50', value: metrics.map_50, key: 'map_50' },
          { name: 'Precision', value: metrics.precision, key: 'precision' },
        ];
      case 'classification':
        return [
          { name: 'Accuracy', value: metrics.accuracy, key: 'accuracy' },
          { name: 'Precision', value: metrics.precision, key: 'precision' },
          { name: 'Recall', value: metrics.recall, key: 'recall' },
        ];
      default:
        return [
          { name: 'Accuracy', value: metrics.accuracy, key: 'accuracy' },
          { name: 'F1 Score', value: metrics.f1_score, key: 'f1_score' },
          { name: 'Precision', value: metrics.precision, key: 'precision' },
        ];
    }
  })();

  return (
    <div
      className={`
        relative bg-white dark:bg-gray-800 rounded-lg border-2 transition-all duration-200 cursor-pointer
        ${isSelected
          ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800 shadow-lg'
          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-md'
        }
        ${className}
      `}
      onClick={handleSelect}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => {
        setShowActions(false);
        setShowDeleteConfirm(false);
      }}
    >
      {/* Status Badge */}
      <div className={`absolute top-3 right-3 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor()}`}>
        <span className="mr-1">{getStatusIcon()}</span>
        {evaluation.status.charAt(0).toUpperCase() + evaluation.status.slice(1)}
      </div>

      {/* Selection Indicator */}
      {isSelected && (
        <div className="absolute top-3 left-3 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      )}

      <div className="p-6">
        {/* Header */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-1">
            {evaluation.name}
          </h3>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            <span className="font-medium">{evaluation.model_name}</span> on{' '}
            <span className="font-medium">{evaluation.dataset_name}</span>
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
            {evaluation.evaluation_type.charAt(0).toUpperCase() + evaluation.evaluation_type.slice(1)} • {' '}
            Created {new Date(evaluation.created_at).toLocaleDateString()}
          </div>
        </div>

        {/* Progress Bar (for running evaluations) */}
        {evaluation.status === 'running' && (
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Progress
              </span>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {Math.round(evaluation.progress * 100)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${evaluation.progress * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Primary Metrics */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          {primaryMetrics.map((metric) => (
            <div key={metric.name} className="text-center">
              <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                {metric.value !== undefined ? formatMetricValue(metric.value, metric.key) : 'N/A'}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {metric.name}
              </div>
            </div>
          ))}
        </div>

        {/* Performance Stats */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-700 dark:text-gray-300">Speed:</span>
            <div className="text-gray-600 dark:text-gray-400">
              {evaluation.metrics.fps 
                ? `${Math.round(evaluation.metrics.fps)} FPS`
                : evaluation.metrics.inference_time_mean
                ? formatMetricValue(evaluation.metrics.inference_time_mean, 'inference_time_mean')
                : 'N/A'
              }
            </div>
          </div>

          <div>
            <span className="font-medium text-gray-700 dark:text-gray-300">Images:</span>
            <div className="text-gray-600 dark:text-gray-400">
              {evaluation.inference_stats.total_images.toLocaleString()}
            </div>
          </div>
        </div>

        {/* Completion Time */}
        {evaluation.completed_at && (
          <div className="mt-4 text-sm">
            <span className="font-medium text-gray-700 dark:text-gray-300">Completed:</span>
            <div className="text-gray-600 dark:text-gray-400">
              {new Date(evaluation.completed_at).toLocaleString()}
              {evaluation.created_at && (
                <span className="ml-2">
                  ({formatTime((new Date(evaluation.completed_at).getTime() - new Date(evaluation.created_at).getTime()) / 1000)})
                </span>
              )}
            </div>
          </div>
        )}

        {/* Class Count */}
        <div className="mt-4 text-sm">
          <span className="font-medium text-gray-700 dark:text-gray-300">Classes:</span>
          <span className="ml-2 text-gray-600 dark:text-gray-400">
            {evaluation.class_metrics.length}
          </span>
        </div>

        {/* Actions Menu */}
        {(showActions || showDeleteConfirm) && (
          <div className="absolute top-12 right-3 bg-white dark:bg-gray-700 rounded-lg shadow-lg border border-gray-200 dark:border-gray-600 py-1 z-10">
            {!showDeleteConfirm ? (
              <>
                {onExport && evaluation.status === 'completed' && (
                  <>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleExport('json');
                      }}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600"
                    >
                      Export JSON
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleExport('csv');
                      }}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600"
                    >
                      Export CSV
                    </button>
                    <hr className="my-1 border-gray-200 dark:border-gray-600" />
                  </>
                )}
                {onDelete && evaluation.status !== 'running' && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowDeleteConfirm(true);
                    }}
                    className="block w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950"
                  >
                    Delete
                  </button>
                )}
              </>
            ) : (
              <div className="px-4 py-3">
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                  Delete this evaluation?
                </p>
                <div className="flex space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowDeleteConfirm(false);
                    }}
                    className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 dark:bg-gray-600 dark:hover:bg-gray-500 rounded"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete();
                    }}
                    className="px-3 py-1 text-xs bg-red-500 hover:bg-red-600 text-white rounded"
                  >
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};