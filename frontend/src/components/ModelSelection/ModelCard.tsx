'use client';

import React, { useState } from 'react';
import { ModelCardProps } from './types';
import { getModelArchitectureInfo, getModelVariantInfo, formatInferenceTime, calculateFPS } from './utils';

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  isSelected = false,
  onSelect,
  onDelete,
  showPerformance = true,
  isReadOnly = false,
  className = '',
}) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const archInfo = getModelArchitectureInfo(model.architecture);
  const variantInfo = getModelVariantInfo(model.architecture, model.variant);

  const handleSelect = () => {
    onSelect?.(model.id);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (showDeleteConfirm) {
      onDelete?.(model.id);
      setShowDeleteConfirm(false);
    } else {
      setShowDeleteConfirm(true);
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'detection':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'segmentation':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getArchitectureIcon = (architecture: string) => {
    switch (architecture) {
      case 'yolo11':
      case 'yolo12':
        return '⚡';
      case 'rtdetr':
        return '🎯';
      case 'sam2':
        return '✂️';
      default:
        return '🤖';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) {
      return 'Today';
    } else if (diffDays === 2) {
      return 'Yesterday';
    } else if (diffDays <= 7) {
      return `${diffDays - 1} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div
      className={`
        relative bg-white dark:bg-gray-800 rounded-lg border-2 transition-all duration-200 cursor-pointer group
        ${isSelected
          ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800 shadow-lg'
          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-md'
        }
        ${className}
      `}
      onClick={handleSelect}
    >
      {/* Selection Indicator */}
      {isSelected && (
        <div className="absolute top-3 left-3 z-10 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      )}

      {/* Delete Button */}
      {!isReadOnly && onDelete && (
        <div className="absolute top-3 right-3 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleDelete}
            className={`
              w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm transition-colors
              ${showDeleteConfirm
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-black bg-opacity-50 hover:bg-red-500'
              }
            `}
            title={showDeleteConfirm ? 'Click again to confirm' : 'Delete model'}
          >
            {showDeleteConfirm ? '✓' : '×'}
          </button>
        </div>
      )}

      <div className="p-6">
        {/* Header */}
        <div className="mb-4">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-2xl">{getArchitectureIcon(model.architecture)}</span>
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-gray-100 text-lg">
                  {model.name}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {archInfo?.name || model.architecture.toUpperCase()} · {variantInfo?.name || model.variant}
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(model.type)}`}>
              {model.type.charAt(0).toUpperCase() + model.type.slice(1)}
            </span>
            <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
              v{model.version}
            </span>
          </div>
        </div>

        {/* Description */}
        {model.description && (
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
            {model.description}
          </p>
        )}

        {/* Model Specs */}
        <div className="space-y-3 mb-4">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">Input Size:</span>
              <div className="text-gray-600 dark:text-gray-400">
                {model.input_size[0]} × {model.input_size[1]}
              </div>
            </div>
            
            {variantInfo && (
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Parameters:</span>
                <div className="text-gray-600 dark:text-gray-400">
                  {variantInfo.params}
                </div>
              </div>
            )}
            
            {variantInfo && (
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Model Size:</span>
                <div className="text-gray-600 dark:text-gray-400">
                  {variantInfo.size}
                </div>
              </div>
            )}

            <div>
              <span className="font-medium text-gray-700 dark:text-gray-300">Categories:</span>
              <div className="text-gray-600 dark:text-gray-400">
                {model.categories.length}
              </div>
            </div>
          </div>

          {/* Supported Formats */}
          <div>
            <span className="font-medium text-gray-700 dark:text-gray-300 text-sm">Formats:</span>
            <div className="flex flex-wrap gap-1 mt-1">
              {model.supported_formats.slice(0, 3).map((format) => (
                <span
                  key={format}
                  className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded"
                >
                  {format.toUpperCase()}
                </span>
              ))}
              {model.supported_formats.length > 3 && (
                <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded">
                  +{model.supported_formats.length - 3}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        {showPerformance && model.performance_metrics && (
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mb-4">
            <h4 className="font-medium text-gray-700 dark:text-gray-300 text-sm mb-3">Performance</h4>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {model.performance_metrics.map && (
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">mAP:</span>
                  <div className="text-green-600 dark:text-green-400 font-medium">
                    {(model.performance_metrics.map * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              
              {model.performance_metrics.map_50 && (
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">mAP@50:</span>
                  <div className="text-green-600 dark:text-green-400 font-medium">
                    {(model.performance_metrics.map_50 * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              
              {model.performance_metrics.inference_time && (
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Speed:</span>
                  <div className="text-blue-600 dark:text-blue-400 font-medium">
                    {formatInferenceTime(model.performance_metrics.inference_time)}
                  </div>
                </div>
              )}
              
              {model.performance_metrics.inference_time && (
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">FPS:</span>
                  <div className="text-blue-600 dark:text-blue-400 font-medium">
                    {calculateFPS(model.performance_metrics.inference_time)}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Created {formatDate(model.created_at)}</span>
            {model.updated_at !== model.created_at && (
              <span>Updated {formatDate(model.updated_at)}</span>
            )}
          </div>

          {/* Quick Actions */}
          <div className="flex items-center justify-between mt-3">
            <div className="flex space-x-2">
              {model.paper_url && (
                <a
                  href={model.paper_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Paper
                </a>
              )}
              {model.model_url && (
                <a
                  href={model.model_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                >
                  Weights
                </a>
              )}
            </div>
            
            <button
              onClick={(e) => {
                e.stopPropagation();
                navigator.clipboard.writeText(model.id);
                // Show toast notification
              }}
              className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
              title="Copy model ID"
            >
              Copy ID
            </button>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Overlay */}
      {showDeleteConfirm && (
        <div className="absolute inset-0 bg-black bg-opacity-75 rounded-lg flex items-center justify-center z-20">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 max-w-xs text-center">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
              Delete Model?
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              This action cannot be undone. All training data and configurations will be lost.
            </p>
            <div className="flex space-x-2 justify-center">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowDeleteConfirm(false);
                }}
                className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 rounded"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="px-3 py-1 text-sm bg-red-500 hover:bg-red-600 text-white rounded"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};