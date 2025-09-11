// T078: Model Selection and Configuration - Type Definitions

export interface ModelInfo {
  id: string;
  name: string;
  type: 'detection' | 'segmentation';
  architecture: 'yolo11' | 'yolo12' | 'rtdetr' | 'sam2';
  variant: string;
  version: string;
  description?: string;
  paper_url?: string;
  model_url?: string;
  config_path?: string;
  weights_path?: string;
  input_size: [number, number];
  supported_formats: string[];
  categories: string[];
  performance_metrics?: {
    map?: number;
    map_50?: number;
    map_75?: number;
    inference_time?: number;
    model_size?: number;
  };
  created_at: string;
  updated_at: string;
}

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  momentum: number;
  warmup_epochs: number;
  lr_scheduler: 'cosine' | 'linear' | 'step';
  optimizer: 'adam' | 'adamw' | 'sgd';
  augmentation: AugmentationConfig;
  early_stopping: {
    enabled: boolean;
    patience: number;
    min_delta: number;
  };
  save_period: number;
  val_period: number;
  device: 'auto' | 'cpu' | 'cuda';
  mixed_precision: boolean;
  gradient_clip: number;
}

export interface AugmentationConfig {
  enabled: boolean;
  hsv_h: number;
  hsv_s: number;
  hsv_v: number;
  degrees: number;
  translate: number;
  scale: number;
  shear: number;
  perspective: number;
  flipud: number;
  fliplr: number;
  mosaic: number;
  mixup: number;
  copy_paste: number;
}

export interface InferenceConfig {
  confidence_threshold: number;
  iou_threshold: number;
  max_detections: number;
  agnostic_nms: boolean;
  multi_label: boolean;
  classes?: number[];
  device: 'auto' | 'cpu' | 'cuda';
  half_precision: boolean;
  batch_size: number;
}

export interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModelId?: string;
  onModelSelect: (modelId: string) => void;
  onModelCreate?: () => void;
  onModelDelete?: (modelId: string) => void;
  filterByType?: 'detection' | 'segmentation' | 'all';
  filterByArchitecture?: 'yolo11' | 'yolo12' | 'rtdetr' | 'sam2' | 'all';
  showPerformance?: boolean;
  isReadOnly?: boolean;
  className?: string;
}

export interface ModelCardProps {
  model: ModelInfo;
  isSelected?: boolean;
  onSelect?: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  showPerformance?: boolean;
  isReadOnly?: boolean;
  className?: string;
}

export interface ModelConfiguratorProps {
  model: ModelInfo;
  config: TrainingConfig | InferenceConfig;
  configType: 'training' | 'inference';
  onConfigChange: (config: TrainingConfig | InferenceConfig) => void;
  onSave?: () => void;
  onReset?: () => void;
  isReadOnly?: boolean;
  className?: string;
}

export interface TrainingConfigFormProps {
  config: TrainingConfig;
  onChange: (config: TrainingConfig) => void;
  modelType: 'detection' | 'segmentation';
  isReadOnly?: boolean;
}

export interface InferenceConfigFormProps {
  config: InferenceConfig;
  onChange: (config: InferenceConfig) => void;
  modelType: 'detection' | 'segmentation';
  isReadOnly?: boolean;
}

export interface ModelMetricsProps {
  model: ModelInfo;
  showComparison?: boolean;
  comparisonModels?: ModelInfo[];
  className?: string;
}

export interface ModelDownloaderProps {
  availableModels: PretrainedModel[];
  onDownload: (model: PretrainedModel) => void;
  downloadProgress?: { [modelId: string]: number };
  className?: string;
}

export interface PretrainedModel {
  id: string;
  name: string;
  architecture: 'yolo11' | 'yolo12' | 'rtdetr' | 'sam2';
  variant: string;
  type: 'detection' | 'segmentation';
  dataset_trained: string;
  download_url: string;
  file_size: number;
  checksum: string;
  performance_metrics: {
    map?: number;
    map_50?: number;
    inference_time?: number;
  };
  description: string;
  license: string;
}

export interface ModelComparisonProps {
  models: ModelInfo[];
  selectedMetrics: ComparisonMetric[];
  onMetricToggle: (metric: ComparisonMetric) => void;
  className?: string;
}

export type ComparisonMetric = 
  | 'map'
  | 'map_50' 
  | 'map_75'
  | 'inference_time'
  | 'model_size'
  | 'parameters';

export interface ModelValidatorProps {
  model: ModelInfo;
  testImages?: string[];
  onValidation: (results: ValidationResult) => void;
  isValidating?: boolean;
  className?: string;
}

export interface ValidationResult {
  model_id: string;
  test_images: number;
  successful_inferences: number;
  failed_inferences: number;
  average_inference_time: number;
  memory_usage: number;
  errors: string[];
  warnings: string[];
  timestamp: string;
}

export interface ModelExportProps {
  model: ModelInfo;
  exportFormats: ExportFormat[];
  onExport: (format: ExportFormat, options: ExportOptions) => void;
  isExporting?: boolean;
  className?: string;
}

export interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  description: string;
  supports_optimization: boolean;
  target_platforms: string[];
}

export interface ExportOptions {
  optimize: boolean;
  quantization?: 'int8' | 'fp16';
  batch_size?: number;
  input_shape?: [number, number];
  include_postprocessing: boolean;
}

export interface HyperparameterTunerProps {
  model: ModelInfo;
  searchSpace: HyperparameterSearchSpace;
  onTuningStart: (config: TuningConfig) => void;
  onTuningComplete?: (results: TuningResult) => void;
  isRunning?: boolean;
  className?: string;
}

export interface HyperparameterSearchSpace {
  learning_rate: { min: number; max: number; scale: 'log' | 'linear' };
  batch_size: number[];
  epochs: { min: number; max: number };
  weight_decay: { min: number; max: number; scale: 'log' | 'linear' };
  momentum: { min: number; max: number };
}

export interface TuningConfig {
  search_method: 'grid' | 'random' | 'bayesian';
  n_trials: number;
  timeout_hours: number;
  early_stopping: boolean;
  metric_to_optimize: 'map' | 'map_50' | 'loss';
  direction: 'maximize' | 'minimize';
}

export interface TuningResult {
  best_params: Record<string, number>;
  best_score: number;
  n_trials: number;
  trials: TuningTrial[];
  optimization_time: number;
}

export interface TuningTrial {
  trial_id: number;
  params: Record<string, number>;
  score: number;
  duration: number;
  status: 'completed' | 'failed' | 'pruned';
}