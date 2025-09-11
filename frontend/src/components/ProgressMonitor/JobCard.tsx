'use client';

import React, { useState } from 'react';
import { JobCardProps, TrainingJob, InferenceJob } from './types';
import { ProgressBar } from './ProgressBar';
import { 
  isTrainingJob, 
  formatDuration, 
  formatTimeRemaining, 
  formatRelativeTime,
  getJobStatusIcon,
  canPauseJob,
  canResumeJob,
  canCancelJob,
  calculateETA,
  formatMetricValue,
  getLatestMetricValue
} from './utils';

export const JobCard: React.FC<JobCardProps> = ({
  job,
  onCancel,
  onPause,
  onResume,
  onDelete,
  onExpand,
  isExpanded = false,
  className = '',
}) => {
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);

  const handleExpand = () => {
    onExpand?.(job.id);
  };

  const handleCancel = () => {
    onCancel?.(job.id);
  };

  const handlePause = () => {
    onPause?.(job.id);
  };

  const handleResume = () => {
    onResume?.(job.id);
  };

  const handleDelete = () => {
    if (showConfirmDelete) {
      onDelete?.(job.id);
      setShowConfirmDelete(false);
    } else {
      setShowConfirmDelete(true);
    }
  };

  const isTraining = isTrainingJob(job);
  const canShowActions = job.status !== 'completed';

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm ${className}`}>
      {/* Main Job Info */}
      <div className="p-6">
        <div className="flex items-start justify-between">
          {/* Job Details */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-3 mb-2">
              <div className="text-xl" title={`${job.status} job`}>
                {getJobStatusIcon(job.status)}
              </div>
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 truncate">
                  {job.name}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {job.model_name} • {isTraining ? 'Training' : 'Inference'} • {formatRelativeTime(job.created_at)}
                </p>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mb-4">
              <ProgressBar
                progress={job.progress}
                status={job.status}
                showPercentage={true}
                showStatus={true}
                animated={job.status === 'running'}
                size="md"
              />
            </div>

            {/* Status Info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Progress:</span>
                <div className="text-gray-600 dark:text-gray-400">
                  {isTraining 
                    ? `${(job as TrainingJob).current_epoch}/${(job as TrainingJob).total_epochs} epochs`
                    : `${(job as InferenceJob).processed_images}/${(job as InferenceJob).total_images} images`
                  }
                </div>
              </div>
              
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Duration:</span>
                <div className="text-gray-600 dark:text-gray-400">
                  {formatDuration(job.elapsed_time)}
                </div>
              </div>
              
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">
                  {job.status === 'running' ? 'ETA:' : 'Status:'}
                </span>
                <div className="text-gray-600 dark:text-gray-400">
                  {job.status === 'running' 
                    ? calculateETA(job)
                    : job.status.charAt(0).toUpperCase() + job.status.slice(1)
                  }
                </div>
              </div>

              {job.status === 'running' && (
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Remaining:</span>
                  <div className="text-gray-600 dark:text-gray-400">
                    {formatTimeRemaining(job.estimated_remaining)}
                  </div>
                </div>
              )}
            </div>

            {/* Key Metrics Preview */}
            {isTraining && job.status === 'running' && (
              <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                {getLatestMetricValue((job as TrainingJob).metrics, 'loss') && (
                  <div>
                    <span className="font-medium text-gray-700 dark:text-gray-300">Loss:</span>
                    <div className="text-gray-600 dark:text-gray-400">
                      {formatMetricValue(getLatestMetricValue((job as TrainingJob).metrics, 'loss')!, 'loss')}
                    </div>
                  </div>
                )}
                
                {getLatestMetricValue((job as TrainingJob).metrics, 'learning_rate') && (
                  <div>
                    <span className="font-medium text-gray-700 dark:text-gray-300">LR:</span>
                    <div className="text-gray-600 dark:text-gray-400">
                      {formatMetricValue(getLatestMetricValue((job as TrainingJob).metrics, 'learning_rate')!, 'learning_rate')}
                    </div>
                  </div>
                )}

                {getLatestMetricValue((job as TrainingJob).metrics, 'map') && (
                  <div>
                    <span className="font-medium text-gray-700 dark:text-gray-300">mAP:</span>
                    <div className="text-gray-600 dark:text-gray-400">
                      {formatMetricValue(getLatestMetricValue((job as TrainingJob).metrics, 'map')!, 'map')}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Error Message */}
            {job.error_message && (
              <div className="mt-4 p-3 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded">
                <div className="flex items-start space-x-2">
                  <span className="text-red-500 text-sm">🚫</span>
                  <div>
                    <h4 className="text-sm font-medium text-red-800 dark:text-red-200">Error:</h4>
                    <p className="text-sm text-red-700 dark:text-red-300 mt-1">{job.error_message}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-start space-x-2 ml-4">
            {canShowActions && (
              <>
                {canPauseJob(job) && (
                  <button
                    onClick={handlePause}
                    className="p-2 text-gray-400 hover:text-yellow-600 dark:hover:text-yellow-400"
                    title="Pause job"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                  </button>
                )}

                {canResumeJob(job) && (
                  <button
                    onClick={handleResume}
                    className="p-2 text-gray-400 hover:text-green-600 dark:hover:text-green-400"
                    title="Resume job"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                    </svg>
                  </button>
                )}

                {canCancelJob(job) && (
                  <button
                    onClick={handleCancel}
                    className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                    title="Cancel job"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </button>
                )}
              </>
            )}

            {/* Delete Button (for completed/failed jobs) */}
            {onDelete && ['completed', 'failed', 'cancelled'].includes(job.status) && (
              <button
                onClick={handleDelete}
                className={`p-2 ${showConfirmDelete 
                  ? 'text-red-600 dark:text-red-400' 
                  : 'text-gray-400 hover:text-red-600 dark:hover:text-red-400'
                }`}
                title={showConfirmDelete ? 'Click again to confirm deletion' : 'Delete job'}
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" clipRule="evenodd" />
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </button>
            )}

            {/* Expand/Collapse Button */}
            <button
              onClick={handleExpand}
              className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              title={isExpanded ? 'Collapse details' : 'Expand details'}
            >
              <svg
                className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-700">
          <div className="p-6 space-y-6">
            {/* Job Configuration */}
            <div>
              <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                Configuration
              </h4>
              <div className="bg-gray-50 dark:bg-gray-750 rounded p-3">
                <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-x-auto">
                  {JSON.stringify(job.config, null, 2)}
                </pre>
              </div>
            </div>

            {/* Training Metrics Chart Placeholder */}
            {isTraining && (job as TrainingJob).metrics.loss.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Training Metrics
                </h4>
                <div className="bg-gray-50 dark:bg-gray-750 rounded p-4 h-48 flex items-center justify-center text-gray-500 dark:text-gray-400">
                  {/* Placeholder for metrics chart */}
                  <div className="text-center">
                    <svg className="w-8 h-8 mx-auto mb-2" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
                    </svg>
                    <p className="text-sm">Metrics chart would go here</p>
                  </div>
                </div>
              </div>
            )}

            {/* Recent Logs */}
            {job.logs.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Recent Logs ({job.logs.length})
                </h4>
                <div className="bg-gray-900 dark:bg-black rounded p-3 max-h-48 overflow-y-auto">
                  <div className="space-y-1 text-xs font-mono">
                    {job.logs.slice(-10).map((log) => (
                      <div key={log.id} className="flex space-x-2">
                        <span className="text-gray-500 whitespace-nowrap">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                        <span className={`uppercase font-medium w-12 ${
                          log.level === 'error' ? 'text-red-400' :
                          log.level === 'warning' ? 'text-yellow-400' :
                          log.level === 'info' ? 'text-blue-400' :
                          'text-gray-400'
                        }`}>
                          {log.level}
                        </span>
                        <span className="text-gray-300 flex-1">
                          {log.message}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Timestamps */}
            <div>
              <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                Timeline
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Created:</span>
                  <span className="text-gray-900 dark:text-gray-100">
                    {new Date(job.created_at).toLocaleString()}
                  </span>
                </div>
                {job.started_at && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Started:</span>
                    <span className="text-gray-900 dark:text-gray-100">
                      {new Date(job.started_at).toLocaleString()}
                    </span>
                  </div>
                )}
                {job.completed_at && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Completed:</span>
                    <span className="text-gray-900 dark:text-gray-100">
                      {new Date(job.completed_at).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation */}
      {showConfirmDelete && (
        <div className="border-t border-gray-200 dark:border-gray-700 bg-red-50 dark:bg-red-950 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-red-800 dark:text-red-200">
                Delete this job?
              </h4>
              <p className="text-sm text-red-600 dark:text-red-400">
                This action cannot be undone. All logs and results will be lost.
              </p>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setShowConfirmDelete(false)}
                className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 dark:bg-gray-600 dark:hover:bg-gray-500 rounded"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="px-3 py-1 text-sm bg-red-600 hover:bg-red-700 text-white rounded"
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