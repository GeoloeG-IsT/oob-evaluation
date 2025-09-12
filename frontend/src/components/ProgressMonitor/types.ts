// T079: Real-time Progress Monitoring - Type Definitions

export interface TrainingJob {
  id: string;
  name: string;
  model_id: string;
  model_name: string;
  status: JobStatus;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  elapsed_time: number;
  estimated_remaining: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  config: TrainingJobConfig;
  metrics: TrainingMetrics;
  logs: LogEntry[];
}

export interface InferenceJob {
  id: string;
  name: string;
  model_id: string;
  model_name: string;
  status: JobStatus;
  progress: number;
  processed_images: number;
  total_images: number;
  current_batch: number;
  total_batches: number;
  elapsed_time: number;
  estimated_remaining: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  config: InferenceJobConfig;
  results: InferenceResults;
  logs: LogEntry[];
}

export type JobStatus = 
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface TrainingJobConfig {
  dataset_id: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  validation_split: number;
  save_frequency: number;
  early_stopping: boolean;
  augmentation_enabled: boolean;
}

export interface InferenceJobConfig {
  image_ids: string[];
  confidence_threshold: number;
  iou_threshold: number;
  batch_size: number;
  output_format: 'json' | 'csv' | 'coco';
  save_visualizations: boolean;
}

export interface TrainingMetrics {
  loss: number[];
  val_loss: number[];
  learning_rate: number[];
  accuracy?: number[];
  val_accuracy?: number[];
  map?: number[];
  val_map?: number[];
  precision?: number[];
  recall?: number[];
  f1_score?: number[];
  epoch_times: number[];
  gpu_memory_usage: number[];
  cpu_usage: number[];
}

export interface InferenceResults {
  total_detections: number;
  detections_per_image: number[];
  confidence_distribution: number[];
  processing_times: number[];
  memory_usage: number[];
  annotations_created: number;
  output_files: string[];
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  details?: Record<string, unknown>;
}

export interface ProgressMonitorProps {
  jobs: (TrainingJob | InferenceJob)[];
  onJobCancel?: (jobId: string) => void;
  onJobPause?: (jobId: string) => void;
  onJobResume?: (jobId: string) => void;
  onJobDelete?: (jobId: string) => void;
  autoRefreshInterval?: number;
  maxVisibleJobs?: number;
  className?: string;
}

export interface JobCardProps {
  job: TrainingJob | InferenceJob;
  onCancel?: (jobId: string) => void;
  onPause?: (jobId: string) => void;
  onResume?: (jobId: string) => void;
  onDelete?: (jobId: string) => void;
  onExpand?: (jobId: string) => void;
  isExpanded?: boolean;
  className?: string;
}

export interface ProgressBarProps {
  progress: number;
  status: JobStatus;
  showPercentage?: boolean;
  showStatus?: boolean;
  size?: 'sm' | 'md' | 'lg';
  animated?: boolean;
  className?: string;
}

export interface MetricsChartProps {
  metrics: TrainingMetrics;
  selectedMetrics: string[];
  onMetricToggle?: (metric: string) => void;
  timeRange?: 'all' | 'last_hour' | 'last_10_epochs';
  className?: string;
}

export interface LogViewerProps {
  logs: LogEntry[];
  maxLines?: number;
  autoScroll?: boolean;
  filterLevel?: LogEntry['level'];
  searchTerm?: string;
  className?: string;
}

export interface JobQueueProps {
  pendingJobs: (TrainingJob | InferenceJob)[];
  runningJobs: (TrainingJob | InferenceJob)[];
  completedJobs: (TrainingJob | InferenceJob)[];
  onJobRequeue?: (jobId: string) => void;
  onJobCancel?: (jobId: string) => void;
  onQueueClear?: () => void;
  maxConcurrentJobs?: number;
  className?: string;
}

export interface ResourceMonitorProps {
  systemMetrics: SystemMetrics;
  refreshInterval?: number;
  showHistory?: boolean;
  className?: string;
}

export interface SystemMetrics {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  memory_total: number;
  gpu_usage?: number[];
  gpu_memory_usage?: number[];
  gpu_memory_total?: number[];
  gpu_temperature?: number[];
  disk_usage: number;
  disk_total: number;
  network_io: {
    bytes_sent: number;
    bytes_recv: number;
  };
}

export interface NotificationProps {
  notifications: Notification[];
  onDismiss?: (notificationId: string) => void;
  onClearAll?: () => void;
  maxVisible?: number;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  className?: string;
}

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  jobId?: string;
  autoHide?: boolean;
  hideAfter?: number; // milliseconds
}

export interface JobComparisonProps {
  jobs: (TrainingJob | InferenceJob)[];
  selectedJobs: string[];
  onJobSelect?: (jobId: string) => void;
  comparisonMetrics: string[];
  onMetricToggle?: (metric: string) => void;
  className?: string;
}

export interface ExportReportProps {
  job: TrainingJob | InferenceJob;
  exportFormats: ReportFormat[];
  onExport?: (format: ReportFormat, options: ExportOptions) => void;
  isExporting?: boolean;
  className?: string;
}

export interface ReportFormat {
  id: string;
  name: string;
  extension: string;
  description: string;
  supports_charts: boolean;
  supports_logs: boolean;
}

export interface ExportOptions {
  include_charts: boolean;
  include_logs: boolean;
  include_config: boolean;
  log_level_filter?: LogEntry['level'];
  date_range?: {
    start: string;
    end: string;
  };
}

export interface RealTimeUpdaterProps {
  jobIds: string[];
  onJobUpdate?: (job: TrainingJob | InferenceJob) => void;
  onSystemUpdate?: (metrics: SystemMetrics) => void;
  onNotification?: (notification: Notification) => void;
  updateInterval?: number;
  reconnectInterval?: number;
  className?: string;
}

export interface WebSocketMessage {
  type: 'job_update' | 'system_metrics' | 'notification' | 'log_entry';
  payload: unknown;
  timestamp: string;
}

export interface JobFiltersProps {
  statusFilter: JobStatus | 'all';
  typeFilter: 'training' | 'inference' | 'all';
  dateRange: {
    start: string;
    end: string;
  } | null;
  onStatusFilterChange: (status: JobStatus | 'all') => void;
  onTypeFilterChange: (type: 'training' | 'inference' | 'all') => void;
  onDateRangeChange: (range: { start: string; end: string } | null) => void;
  onClearFilters: () => void;
  className?: string;
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  condition: AlertCondition;
  actions: AlertAction[];
  created_at: string;
  updated_at: string;
}

export interface AlertCondition {
  type: 'metric_threshold' | 'job_status' | 'time_based' | 'error_pattern';
  metric?: string;
  threshold?: number;
  operator?: 'gt' | 'lt' | 'eq' | 'ne';
  status?: JobStatus;
  duration?: number; // minutes
  pattern?: string; // regex for error patterns
}

export interface AlertAction {
  type: 'notification' | 'email' | 'webhook' | 'pause_job' | 'stop_job';
  config: Record<string, unknown>;
}

export interface AlertManagerProps {
  rules: AlertRule[];
  onRuleCreate?: (rule: Omit<AlertRule, 'id' | 'created_at' | 'updated_at'>) => void;
  onRuleUpdate?: (ruleId: string, updates: Partial<AlertRule>) => void;
  onRuleDelete?: (ruleId: string) => void;
  onRuleToggle?: (ruleId: string, enabled: boolean) => void;
  activeAlerts: ActiveAlert[];
  className?: string;
}

export interface ActiveAlert {
  id: string;
  rule_id: string;
  rule_name: string;
  triggered_at: string;
  resolved_at?: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  job_id?: string;
  acknowledged: boolean;
}

export interface JobSchedulerProps {
  scheduledJobs: ScheduledJob[];
  onJobSchedule?: (job: Omit<ScheduledJob, 'id' | 'created_at' | 'updated_at'>) => void;
  onJobUnschedule?: (jobId: string) => void;
  onScheduleUpdate?: (jobId: string, schedule: CronSchedule) => void;
  className?: string;
}

export interface ScheduledJob {
  id: string;
  name: string;
  description: string;
  job_type: 'training' | 'inference';
  job_config: TrainingJobConfig | InferenceJobConfig;
  schedule: CronSchedule;
  enabled: boolean;
  next_run: string;
  last_run?: string;
  run_count: number;
  created_at: string;
  updated_at: string;
}

export interface CronSchedule {
  expression: string;
  timezone: string;
  description: string;
}