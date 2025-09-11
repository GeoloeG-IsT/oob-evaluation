// T079: Real-time Progress Monitoring - Main Export

export { ProgressMonitor } from './ProgressMonitor';
export { JobCard } from './JobCard';
export { ProgressBar } from './ProgressBar';
export { JobFilters } from './JobFilters';

export type * from './types';

export {
  formatDuration,
  formatTimeRemaining,
  formatRelativeTime,
  isTrainingJob,
  getJobProgress,
  getJobType,
  getJobStatusColor,
  getJobStatusIcon,
  canPauseJob,
  canResumeJob,
  canCancelJob,
  estimateTimeRemaining,
  calculateETA,
  getLatestMetricValue,
  calculateMetricTrend,
  formatMetricValue,
  formatBytes,
  formatPercentage,
  getResourceUsageLevel,
  getResourceUsageColor,
  getLogLevelColor,
  getLogLevelIcon,
  filterLogsByLevel,
  searchLogs,
  sortJobsByStatus,
  createWebSocketUrl,
  shouldReconnectWebSocket,
  generateJobReport,
  fetchJobStatus,
  cancelJob,
  pauseJob,
  resumeJob,
  deleteJob,
  fetchSystemMetrics,
  fetchJobLogs,
} from './utils';