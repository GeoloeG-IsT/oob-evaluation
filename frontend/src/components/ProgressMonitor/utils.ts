// T079: Real-time Progress Monitoring - Utility Functions

import { TrainingJob, InferenceJob, JobStatus, TrainingMetrics, SystemMetrics, LogEntry } from './types';

// Time formatting utilities
export const formatDuration = (milliseconds: number): string => {
  if (milliseconds < 1000) {
    return `${milliseconds}ms`;
  }
  
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days}d ${hours % 24}h ${minutes % 60}m`;
  } else if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
};

export const formatTimeRemaining = (milliseconds: number): string => {
  if (milliseconds <= 0) return 'Complete';
  if (milliseconds === Infinity) return 'Unknown';
  
  return `~${formatDuration(milliseconds)} remaining`;
};

export const formatRelativeTime = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  
  if (diffMs < 60000) { // Less than 1 minute
    return 'Just now';
  } else if (diffMs < 3600000) { // Less than 1 hour
    const minutes = Math.floor(diffMs / 60000);
    return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
  } else if (diffMs < 86400000) { // Less than 1 day
    const hours = Math.floor(diffMs / 3600000);
    return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
  } else {
    return date.toLocaleDateString();
  }
};

// Job utilities
export const isTrainingJob = (job: TrainingJob | InferenceJob): job is TrainingJob => {
  return 'total_epochs' in job;
};

export const getJobProgress = (job: TrainingJob | InferenceJob): number => {
  return job.progress || 0;
};

export const getJobType = (job: TrainingJob | InferenceJob): 'training' | 'inference' => {
  return isTrainingJob(job) ? 'training' : 'inference';
};

export const getJobStatusColor = (status: JobStatus): string => {
  switch (status) {
    case 'pending':
      return 'bg-gray-500';
    case 'running':
      return 'bg-blue-500';
    case 'paused':
      return 'bg-yellow-500';
    case 'completed':
      return 'bg-green-500';
    case 'failed':
      return 'bg-red-500';
    case 'cancelled':
      return 'bg-gray-400';
    default:
      return 'bg-gray-500';
  }
};

export const getJobStatusIcon = (status: JobStatus): string => {
  switch (status) {
    case 'pending':
      return '⏳';
    case 'running':
      return '▶️';
    case 'paused':
      return '⏸️';
    case 'completed':
      return '✅';
    case 'failed':
      return '❌';
    case 'cancelled':
      return '⏹️';
    default:
      return '❓';
  }
};

export const canPauseJob = (job: TrainingJob | InferenceJob): boolean => {
  return job.status === 'running' && isTrainingJob(job);
};

export const canResumeJob = (job: TrainingJob | InferenceJob): boolean => {
  return job.status === 'paused';
};

export const canCancelJob = (job: TrainingJob | InferenceJob): boolean => {
  return ['pending', 'running', 'paused'].includes(job.status);
};

export const estimateTimeRemaining = (
  currentProgress: number,
  elapsedTime: number
): number => {
  if (currentProgress <= 0 || currentProgress >= 1) {
    return 0;
  }
  
  const rate = currentProgress / elapsedTime;
  const remainingProgress = 1 - currentProgress;
  return remainingProgress / rate;
};

export const calculateETA = (job: TrainingJob | InferenceJob): string => {
  if (job.status !== 'running' || job.progress >= 1) {
    return 'N/A';
  }
  
  const remaining = estimateTimeRemaining(job.progress, job.elapsed_time);
  const etaDate = new Date(Date.now() + remaining);
  
  return etaDate.toLocaleTimeString();
};

// Metrics utilities
export const getLatestMetricValue = (
  metrics: TrainingMetrics,
  metricName: keyof TrainingMetrics
): number | null => {
  const values = metrics[metricName];
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  return values[values.length - 1];
};

export const calculateMetricTrend = (
  values: number[],
  windowSize: number = 5
): 'up' | 'down' | 'stable' => {
  if (values.length < windowSize * 2) {
    return 'stable';
  }
  
  const recent = values.slice(-windowSize);
  const previous = values.slice(-(windowSize * 2), -windowSize);
  
  const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length;
  const previousAvg = previous.reduce((sum, val) => sum + val, 0) / previous.length;
  
  const threshold = 0.01; // 1% threshold
  if (recentAvg > previousAvg * (1 + threshold)) {
    return 'up';
  } else if (recentAvg < previousAvg * (1 - threshold)) {
    return 'down';
  } else {
    return 'stable';
  }
};

export const formatMetricValue = (
  value: number,
  metricName: string,
  precision: number = 4
): string => {
  if (metricName.includes('time')) {
    return formatDuration(value);
  } else if (metricName.includes('rate') || metricName.includes('learning')) {
    return value.toExponential(2);
  } else if (metricName.includes('usage') || metricName.includes('accuracy') || metricName.includes('map')) {
    return `${(value * 100).toFixed(1)}%`;
  } else {
    return value.toFixed(precision);
  }
};

// System monitoring utilities
export const formatBytes = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
};

export const formatPercentage = (value: number): string => {
  return `${Math.round(value)}%`;
};

export const getResourceUsageLevel = (usage: number): 'low' | 'medium' | 'high' | 'critical' => {
  if (usage < 0.25) return 'low';
  if (usage < 0.5) return 'medium';
  if (usage < 0.8) return 'high';
  return 'critical';
};

export const getResourceUsageColor = (usage: number): string => {
  const level = getResourceUsageLevel(usage);
  switch (level) {
    case 'low':
      return 'text-green-600 dark:text-green-400';
    case 'medium':
      return 'text-yellow-600 dark:text-yellow-400';
    case 'high':
      return 'text-orange-600 dark:text-orange-400';
    case 'critical':
      return 'text-red-600 dark:text-red-400';
  }
};

// Log utilities
export const getLogLevelColor = (level: LogEntry['level']): string => {
  switch (level) {
    case 'debug':
      return 'text-gray-500 dark:text-gray-400';
    case 'info':
      return 'text-blue-600 dark:text-blue-400';
    case 'warning':
      return 'text-yellow-600 dark:text-yellow-400';
    case 'error':
      return 'text-red-600 dark:text-red-400';
  }
};

export const getLogLevelIcon = (level: LogEntry['level']): string => {
  switch (level) {
    case 'debug':
      return '🐛';
    case 'info':
      return 'ℹ️';
    case 'warning':
      return '⚠️';
    case 'error':
      return '🚫';
  }
};

export const filterLogsByLevel = (
  logs: LogEntry[],
  minLevel: LogEntry['level']
): LogEntry[] => {
  const levelOrder = ['debug', 'info', 'warning', 'error'];
  const minLevelIndex = levelOrder.indexOf(minLevel);
  
  return logs.filter(log => {
    const logLevelIndex = levelOrder.indexOf(log.level);
    return logLevelIndex >= minLevelIndex;
  });
};

export const searchLogs = (logs: LogEntry[], searchTerm: string): LogEntry[] => {
  if (!searchTerm.trim()) return logs;
  
  const term = searchTerm.toLowerCase();
  return logs.filter(log =>
    log.message.toLowerCase().includes(term) ||
    (log.details && JSON.stringify(log.details).toLowerCase().includes(term))
  );
};

// Sorting utilities
export const sortJobsByStatus = (jobs: (TrainingJob | InferenceJob)[]): (TrainingJob | InferenceJob)[] => {
  const statusPriority: Record<JobStatus, number> = {
    'running': 1,
    'paused': 2,
    'pending': 3,
    'failed': 4,
    'cancelled': 5,
    'completed': 6,
  };
  
  return [...jobs].sort((a, b) => {
    const priorityA = statusPriority[a.status] || 999;
    const priorityB = statusPriority[b.status] || 999;
    
    if (priorityA !== priorityB) {
      return priorityA - priorityB;
    }
    
    // Secondary sort by creation date (newest first)
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });
};

// WebSocket utilities
export const createWebSocketUrl = (endpoint: string): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return `${protocol}//${host}${endpoint}`;
};

export const shouldReconnectWebSocket = (
  closeEvent: CloseEvent
): boolean => {
  // Don't reconnect for normal closures or policy violations
  return ![1000, 1001, 1005, 1006].includes(closeEvent.code);
};

// Export utilities
export const generateJobReport = (job: TrainingJob | InferenceJob): string => {
  const report = {
    id: job.id,
    name: job.name,
    type: getJobType(job),
    status: job.status,
    progress: `${(job.progress * 100).toFixed(1)}%`,
    duration: formatDuration(job.elapsed_time),
    created_at: job.created_at,
    started_at: job.started_at,
    completed_at: job.completed_at,
    config: job.config,
  };
  
  if (isTrainingJob(job)) {
    Object.assign(report, {
      epochs: `${job.current_epoch}/${job.total_epochs}`,
      steps: `${job.current_step}/${job.total_steps}`,
      latest_metrics: {
        loss: getLatestMetricValue(job.metrics, 'loss'),
        val_loss: getLatestMetricValue(job.metrics, 'val_loss'),
        map: getLatestMetricValue(job.metrics, 'map'),
      },
    });
  } else {
    Object.assign(report, {
      images: `${job.processed_images}/${job.total_images}`,
      batches: `${job.current_batch}/${job.total_batches}`,
      results: job.results,
    });
  }
  
  return JSON.stringify(report, null, 2);
};

// API utilities
export const fetchJobStatus = async (jobId: string): Promise<TrainingJob | InferenceJob> => {
  const response = await fetch(`/api/v1/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch job status: ${response.statusText}`);
  }
  return response.json();
};

export const cancelJob = async (jobId: string): Promise<void> => {
  const response = await fetch(`/api/v1/jobs/${jobId}/cancel`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Failed to cancel job: ${response.statusText}`);
  }
};

export const pauseJob = async (jobId: string): Promise<void> => {
  const response = await fetch(`/api/v1/jobs/${jobId}/pause`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Failed to pause job: ${response.statusText}`);
  }
};

export const resumeJob = async (jobId: string): Promise<void> => {
  const response = await fetch(`/api/v1/jobs/${jobId}/resume`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Failed to resume job: ${response.statusText}`);
  }
};

export const deleteJob = async (jobId: string): Promise<void> => {
  const response = await fetch(`/api/v1/jobs/${jobId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete job: ${response.statusText}`);
  }
};

export const fetchSystemMetrics = async (): Promise<SystemMetrics> => {
  const response = await fetch('/api/v1/system/metrics');
  if (!response.ok) {
    throw new Error(`Failed to fetch system metrics: ${response.statusText}`);
  }
  return response.json();
};

export const fetchJobLogs = async (jobId: string, limit?: number): Promise<LogEntry[]> => {
  const params = new URLSearchParams();
  if (limit) params.append('limit', limit.toString());
  
  const response = await fetch(`/api/v1/jobs/${jobId}/logs?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch job logs: ${response.statusText}`);
  }
  return response.json();
};