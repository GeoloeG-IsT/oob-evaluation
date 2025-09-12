// T080: Performance Evaluation and Comparison Dashboard - Type Definitions

export interface EvaluationResult {
  id: string;
  name: string;
  model_id: string;
  model_name: string;
  dataset_id: string;
  dataset_name: string;
  evaluation_type: 'detection' | 'segmentation' | 'classification';
  status: 'running' | 'completed' | 'failed';
  progress: number;
  metrics: PerformanceMetrics;
  confusion_matrix?: ConfusionMatrix;
  class_metrics: ClassMetrics[];
  inference_stats: InferenceStats;
  created_at: string;
  completed_at?: string;
  config: EvaluationConfig;
}

export interface PerformanceMetrics {
  // Detection metrics
  map?: number; // Mean Average Precision
  map_50?: number; // mAP at IoU 0.5
  map_75?: number; // mAP at IoU 0.75
  map_small?: number; // mAP for small objects
  map_medium?: number; // mAP for medium objects
  map_large?: number; // mAP for large objects
  
  // Classification metrics
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  
  // Per-class metrics
  precision_macro?: number;
  recall_macro?: number;
  f1_macro?: number;
  precision_weighted?: number;
  recall_weighted?: number;
  f1_weighted?: number;
  
  // Inference performance
  inference_time_mean?: number;
  inference_time_std?: number;
  fps?: number;
  
  // Memory usage
  memory_usage_peak?: number;
  memory_usage_mean?: number;
  
  // Model size metrics
  model_size?: number;
  parameters?: number;
  flops?: number;
}

export interface ClassMetrics {
  class_id: number;
  class_name: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
  ap?: number; // Average Precision for detection
  instances?: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
}

export interface ConfusionMatrix {
  matrix: number[][];
  class_names: string[];
  normalized?: boolean;
}

export interface InferenceStats {
  total_images: number;
  total_detections: number;
  detections_per_image_mean: number;
  detections_per_image_std: number;
  inference_times: number[];
  batch_sizes: number[];
  memory_usage: number[];
  confidence_distribution: {
    bins: number[];
    counts: number[];
  };
}

export interface EvaluationConfig {
  confidence_threshold: number;
  iou_threshold: number;
  max_detections?: number;
  class_filter?: number[];
  save_predictions: boolean;
  save_visualizations: boolean;
  batch_size: number;
}

export interface EvaluationDashboardProps {
  evaluations: EvaluationResult[];
  selectedEvaluations?: string[];
  onEvaluationSelect?: (evaluationId: string) => void;
  onEvaluationDelete?: (evaluationId: string) => void;
  onEvaluationCreate?: () => void;
  onEvaluationExport?: (evaluationId: string, format: string) => void;
  className?: string;
}

export interface MetricsComparisonProps {
  evaluations: EvaluationResult[];
  selectedMetrics: string[];
  onMetricToggle: (metric: string) => void;
  comparisonMode: 'table' | 'chart' | 'radar';
  onComparisonModeChange: (mode: 'table' | 'chart' | 'radar') => void;
  className?: string;
}

export interface PerformanceChartProps {
  evaluations: EvaluationResult[];
  metric: string;
  chartType: 'bar' | 'line' | 'scatter' | 'box';
  groupBy?: 'model' | 'dataset' | 'class' | 'time';
  className?: string;
}

export interface ConfusionMatrixProps {
  confusionMatrix: ConfusionMatrix;
  showNormalized?: boolean;
  showPercentages?: boolean;
  colorScheme?: 'blue' | 'green' | 'red' | 'viridis';
  className?: string;
}

export interface ClassMetricsTableProps {
  classMetrics: ClassMetrics[];
  sortBy?: keyof ClassMetrics;
  sortOrder?: 'asc' | 'desc';
  onSort?: (column: keyof ClassMetrics) => void;
  highlightBest?: boolean;
  highlightWorst?: boolean;
  className?: string;
}

export interface ModelBenchmarkProps {
  evaluations: EvaluationResult[];
  benchmarkMetrics: BenchmarkMetric[];
  onMetricAdd?: (metric: BenchmarkMetric) => void;
  onMetricRemove?: (metricId: string) => void;
  className?: string;
}

export interface BenchmarkMetric {
  id: string;
  name: string;
  display_name: string;
  unit: string;
  higher_is_better: boolean;
  format: 'percentage' | 'decimal' | 'integer' | 'time' | 'bytes';
  category: 'accuracy' | 'speed' | 'efficiency' | 'memory';
}

export interface ModelRankingProps {
  evaluations: EvaluationResult[];
  rankingCriteria: RankingCriterion[];
  onCriteriaChange: (criteria: RankingCriterion[]) => void;
  weightingMethod: 'equal' | 'manual' | 'automatic';
  className?: string;
}

export interface RankingCriterion {
  metric: string;
  weight: number;
  higher_is_better: boolean;
}

export interface EvaluationReportProps {
  evaluation: EvaluationResult;
  includeCharts?: boolean;
  includeConfusionMatrix?: boolean;
  includeClassMetrics?: boolean;
  includeInferenceStats?: boolean;
  exportFormats?: ReportFormat[];
  onExport?: (format: ReportFormat) => void;
  className?: string;
}

export interface ReportFormat {
  id: string;
  name: string;
  extension: string;
  mime_type: string;
  supports_images: boolean;
  supports_interactive: boolean;
}

export interface StatisticalTestProps {
  evaluations: EvaluationResult[];
  metric: string;
  testType: 'ttest' | 'anova' | 'kruskal' | 'friedman';
  significanceLevel: number;
  onTestComplete?: (result: StatisticalTestResult) => void;
  className?: string;
}

export interface StatisticalTestResult {
  test_type: string;
  statistic: number;
  p_value: number;
  significant: boolean;
  confidence_level: number;
  effect_size?: number;
  interpretation: string;
  recommendations: string[];
}

export interface DataExplorationProps {
  evaluation: EvaluationResult;
  explorationMode: 'overview' | 'class_analysis' | 'error_analysis' | 'prediction_analysis';
  onModeChange: (mode: typeof explorationMode) => void;
  className?: string;
}

export interface ErrorAnalysisProps {
  evaluation: EvaluationResult;
  errorTypes: ErrorType[];
  selectedErrorTypes: string[];
  onErrorTypeToggle: (errorType: string) => void;
  className?: string;
}

export interface ErrorType {
  id: string;
  name: string;
  description: string;
  count: number;
  examples: ErrorExample[];
}

export interface ErrorExample {
  image_id: string;
  image_path: string;
  ground_truth: Annotation[];
  predictions: Prediction[];
  error_score: number;
  error_description: string;
}

export interface Annotation {
  class_id: number;
  class_name: string;
  bbox?: [number, number, number, number];
  polygon?: number[][];
  confidence?: number;
}

export interface Prediction {
  class_id: number;
  class_name: string;
  bbox?: [number, number, number, number];
  polygon?: number[][];
  confidence: number;
  is_correct: boolean;
  iou_with_gt?: number;
}

export interface MetricTrendProps {
  evaluations: EvaluationResult[];
  metric: string;
  timeframe: 'daily' | 'weekly' | 'monthly';
  showTrendline?: boolean;
  showConfidenceInterval?: boolean;
  className?: string;
}

export interface CalibrationPlotProps {
  evaluation: EvaluationResult;
  numBins?: number;
  showPerfectCalibration?: boolean;
  showStatistics?: boolean;
  className?: string;
}

export interface PrecisionRecallCurveProps {
  evaluation: EvaluationResult;
  classFilter?: number[];
  showAUC?: boolean;
  showIsoF1Curves?: boolean;
  className?: string;
}

export interface ROCCurveProps {
  evaluation: EvaluationResult;
  classFilter?: number[];
  showAUC?: boolean;
  showRandomClassifier?: boolean;
  className?: string;
}

export interface LearningCurveProps {
  evaluations: EvaluationResult[];
  metric: string;
  trainingData: TrainingDataPoint[];
  showValidationCurve?: boolean;
  className?: string;
}

export interface TrainingDataPoint {
  training_size: number;
  metric_value: number;
  validation_value?: number;
  timestamp: string;
}

export interface HyperparameterAnalysisProps {
  evaluations: EvaluationResult[];
  parameters: string[];
  targetMetric: string;
  analysisType: 'correlation' | 'importance' | 'interaction';
  className?: string;
}

export interface AlertConfiguration {
  id: string;
  name: string;
  metric: string;
  condition: 'above' | 'below' | 'between' | 'change';
  threshold: number | [number, number];
  enabled: boolean;
  notification_channels: string[];
}

export interface ModelComparisonSummary {
  winner: string;
  improvements: {
    metric: string;
    improvement: number;
    significance: boolean;
  }[];
  trade_offs: {
    metric: string;
    degradation: number;
  }[];
  recommendation: string;
}