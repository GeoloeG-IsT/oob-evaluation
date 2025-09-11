'use client';

import React from 'react';
import { ProgressBarProps } from './types';
import { getJobStatusColor } from './utils';

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  status,
  showPercentage = true,
  showStatus = false,
  size = 'md',
  animated = false,
  className = '',
}) => {
  const percentage = Math.min(Math.max(progress * 100, 0), 100);
  
  const sizeClasses = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4',
  };

  const textSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  };

  const getStatusText = () => {
    switch (status) {
      case 'pending':
        return 'Pending';
      case 'running':
        return 'Running';
      case 'paused':
        return 'Paused';
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      case 'cancelled':
        return 'Cancelled';
      default:
        return status;
    }
  };

  return (
    <div className={`progress-bar ${className}`}>
      {/* Labels */}
      {(showPercentage || showStatus) && (
        <div className={`flex items-center justify-between mb-2 ${textSizeClasses[size]}`}>
          {showStatus && (
            <span className="font-medium text-gray-700 dark:text-gray-300">
              {getStatusText()}
            </span>
          )}
          {showPercentage && (
            <span className="text-gray-600 dark:text-gray-400">
              {percentage.toFixed(1)}%
            </span>
          )}
        </div>
      )}

      {/* Progress Bar */}
      <div className={`relative w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden ${sizeClasses[size]}`}>
        <div
          className={`
            h-full rounded-full transition-all duration-300 ease-out
            ${getJobStatusColor(status)}
            ${animated && status === 'running' ? 'animate-pulse' : ''}
          `}
          style={{
            width: `${percentage}%`,
            background: status === 'running' && animated
              ? `linear-gradient(90deg, ${getStatusColor(status)} 0%, ${getStatusColor(status)}aa 50%, ${getStatusColor(status)} 100%)`
              : undefined,
          }}
        >
          {/* Animated shine effect for running jobs */}
          {animated && status === 'running' && percentage > 0 && (
            <div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-ping"
              style={{
                animationDuration: '2s',
                animationIterationCount: 'infinite',
              }}
            />
          )}
        </div>

        {/* Indeterminate progress for pending status */}
        {status === 'pending' && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-gray-400 to-transparent opacity-50 animate-pulse" />
        )}
      </div>

      {/* Additional status indicators */}
      {size === 'lg' && (
        <div className="mt-2 flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400">
          <div className={`w-2 h-2 rounded-full ${getJobStatusColor(status)}`} />
          <span className="capitalize">{status}</span>
          {status === 'running' && animated && (
            <span className="animate-pulse">●</span>
          )}
        </div>
      )}
    </div>
  );
};

// Helper function to get status colors for backgrounds
const getStatusColor = (status: string): string => {
  switch (status) {
    case 'pending':
      return '#6b7280'; // gray-500
    case 'running':
      return '#3b82f6'; // blue-500
    case 'paused':
      return '#eab308'; // yellow-500
    case 'completed':
      return '#22c55e'; // green-500
    case 'failed':
      return '#ef4444'; // red-500
    case 'cancelled':
      return '#9ca3af'; // gray-400
    default:
      return '#6b7280'; // gray-500
  }
};