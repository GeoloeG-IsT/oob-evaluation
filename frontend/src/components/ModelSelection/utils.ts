// T078: Model Selection and Configuration - Utility Functions

import { ModelInfo, TrainingConfig, InferenceConfig, AugmentationConfig, PretrainedModel } from './types';

// Default configurations
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 100,
  batch_size: 16,
  learning_rate: 0.01,
  weight_decay: 0.0005,
  momentum: 0.937,
  warmup_epochs: 3,
  lr_scheduler: 'cosine',
  optimizer: 'sgd',
  augmentation: {
    enabled: true,
    hsv_h: 0.015,
    hsv_s: 0.7,
    hsv_v: 0.4,
    degrees: 0.0,
    translate: 0.1,
    scale: 0.5,
    shear: 0.0,
    perspective: 0.0,
    flipud: 0.0,
    fliplr: 0.5,
    mosaic: 1.0,
    mixup: 0.0,
    copy_paste: 0.0,
  },
  early_stopping: {
    enabled: true,
    patience: 50,
    min_delta: 0.001,
  },
  save_period: 10,
  val_period: 1,
  device: 'auto',
  mixed_precision: true,
  gradient_clip: 10.0,
};

export const DEFAULT_INFERENCE_CONFIG: InferenceConfig = {
  confidence_threshold: 0.25,
  iou_threshold: 0.45,
  max_detections: 1000,
  agnostic_nms: false,
  multi_label: false,
  device: 'auto',
  half_precision: false,
  batch_size: 1,
};

export const DEFAULT_AUGMENTATION_CONFIG: AugmentationConfig = {
  enabled: true,
  hsv_h: 0.015,
  hsv_s: 0.7,
  hsv_v: 0.4,
  degrees: 0.0,
  translate: 0.1,
  scale: 0.5,
  shear: 0.0,
  perspective: 0.0,
  flipud: 0.0,
  fliplr: 0.5,
  mosaic: 1.0,
  mixup: 0.0,
  copy_paste: 0.0,
};

// Model architecture information
export const MODEL_ARCHITECTURES = {
  yolo11: {
    name: 'YOLO11',
    variants: {
      'yolo11n': { name: 'Nano', params: '2.6M', size: '6.2MB' },
      'yolo11s': { name: 'Small', params: '9.4M', size: '21.5MB' },
      'yolo11m': { name: 'Medium', params: '20.1M', size: '49.7MB' },
      'yolo11l': { name: 'Large', params: '25.3M', size: '86.9MB' },
      'yolo11x': { name: 'Extra Large', params: '56.9M', size: '196.9MB' },
    },
    type: 'detection',
    description: 'Latest YOLO architecture with improved accuracy and speed',
  },
  yolo12: {
    name: 'YOLO12',
    variants: {
      'yolo12n': { name: 'Nano', params: '2.8M', size: '6.5MB' },
      'yolo12s': { name: 'Small', params: '9.8M', size: '22.1MB' },
      'yolo12m': { name: 'Medium', params: '20.8M', size: '51.2MB' },
      'yolo12l': { name: 'Large', params: '26.1M', size: '89.3MB' },
      'yolo12x': { name: 'Extra Large', params: '58.2M', size: '201.4MB' },
    },
    type: 'detection',
    description: 'Next-generation YOLO with enhanced performance',
  },
  rtdetr: {
    name: 'RT-DETR',
    variants: {
      'rtdetr-r18': { name: 'ResNet-18', params: '20M', size: '47MB' },
      'rtdetr-r34': { name: 'ResNet-34', params: '31M', size: '70MB' },
      'rtdetr-r50': { name: 'ResNet-50', params: '42M', size: '108MB' },
      'rtdetr-r101': { name: 'ResNet-101', params: '76M', size: '188MB' },
      'rtdetr-nano': { name: 'RT-DETR Nano', params: '2.5M', size: '6MB' },
      'rtdetr-small': { name: 'RT-DETR Small', params: '8.9M', size: '20MB' },
      'rtdetr-medium': { name: 'RT-DETR Medium', params: '18.8M', size: '43MB' },
    },
    type: 'detection',
    description: 'Real-time DETR detector with transformer architecture',
  },
  sam2: {
    name: 'SAM2',
    variants: {
      'sam2-tiny': { name: 'Tiny', params: '38.9M', size: '147MB' },
      'sam2-small': { name: 'Small', params: '46M', size: '179MB' },
      'sam2-base-plus': { name: 'Base+', params: '110M', size: '420MB' },
      'sam2-large': { name: 'Large', params: '224M', size: '850MB' },
    },
    type: 'segmentation',
    description: 'Segment Anything Model 2 for universal segmentation',
  },
} as const;

// Configuration validation
export const validateTrainingConfig = (config: TrainingConfig): { isValid: boolean; errors: string[] } => {
  const errors: string[] = [];

  if (config.epochs < 1 || config.epochs > 10000) {
    errors.push('Epochs must be between 1 and 10000');
  }

  if (config.batch_size < 1 || config.batch_size > 256) {
    errors.push('Batch size must be between 1 and 256');
  }

  if (config.learning_rate <= 0 || config.learning_rate > 1) {
    errors.push('Learning rate must be between 0 and 1');
  }

  if (config.weight_decay < 0 || config.weight_decay > 0.1) {
    errors.push('Weight decay must be between 0 and 0.1');
  }

  if (config.momentum < 0 || config.momentum > 1) {
    errors.push('Momentum must be between 0 and 1');
  }

  if (config.warmup_epochs < 0 || config.warmup_epochs > config.epochs / 2) {
    errors.push('Warmup epochs must be between 0 and half of total epochs');
  }

  if (config.gradient_clip <= 0 || config.gradient_clip > 100) {
    errors.push('Gradient clip must be between 0 and 100');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};

export const validateInferenceConfig = (config: InferenceConfig): { isValid: boolean; errors: string[] } => {
  const errors: string[] = [];

  if (config.confidence_threshold < 0 || config.confidence_threshold > 1) {
    errors.push('Confidence threshold must be between 0 and 1');
  }

  if (config.iou_threshold < 0 || config.iou_threshold > 1) {
    errors.push('IoU threshold must be between 0 and 1');
  }

  if (config.max_detections < 1 || config.max_detections > 10000) {
    errors.push('Max detections must be between 1 and 10000');
  }

  if (config.batch_size < 1 || config.batch_size > 64) {
    errors.push('Batch size must be between 1 and 64');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};

// Model utilities
export const getModelArchitectureInfo = (architecture: string) => {
  return MODEL_ARCHITECTURES[architecture as keyof typeof MODEL_ARCHITECTURES];
};

export const getModelVariantInfo = (architecture: string, variant: string) => {
  const archInfo = getModelArchitectureInfo(architecture);
  return archInfo?.variants[variant as keyof typeof archInfo.variants];
};

export const formatModelSize = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
};

export const formatInferenceTime = (milliseconds: number): string => {
  if (milliseconds < 1) {
    return `${(milliseconds * 1000).toFixed(0)}μs`;
  } else if (milliseconds < 1000) {
    return `${milliseconds.toFixed(1)}ms`;
  } else {
    return `${(milliseconds / 1000).toFixed(2)}s`;
  }
};

export const calculateFPS = (inferenceTime: number): number => {
  return Math.round(1000 / inferenceTime);
};

export const getModelComplexityScore = (model: ModelInfo): number => {
  const archInfo = getModelArchitectureInfo(model.architecture);
  const variantInfo = getModelVariantInfo(model.architecture, model.variant);
  
  if (!archInfo || !variantInfo) return 0;
  
  // Parse parameters (e.g., "2.6M" -> 2.6)
  const paramStr = variantInfo.params;
  const paramNum = parseFloat(paramStr);
  const paramMultiplier = paramStr.includes('M') ? 1000000 : paramStr.includes('K') ? 1000 : 1;
  const totalParams = paramNum * paramMultiplier;
  
  // Normalize to 0-100 scale (100M params = 100 complexity)
  return Math.min(100, (totalParams / 1000000) * 10);
};

// Configuration presets
export const TRAINING_PRESETS = {
  fast: {
    name: 'Fast Training',
    description: 'Quick training with basic augmentation',
    config: {
      ...DEFAULT_TRAINING_CONFIG,
      epochs: 50,
      batch_size: 32,
      learning_rate: 0.02,
      augmentation: {
        ...DEFAULT_AUGMENTATION_CONFIG,
        mosaic: 0.5,
        mixup: 0.0,
      },
    },
  },
  balanced: {
    name: 'Balanced',
    description: 'Good balance between speed and accuracy',
    config: DEFAULT_TRAINING_CONFIG,
  },
  high_quality: {
    name: 'High Quality',
    description: 'Maximum accuracy with extensive augmentation',
    config: {
      ...DEFAULT_TRAINING_CONFIG,
      epochs: 300,
      batch_size: 8,
      learning_rate: 0.005,
      augmentation: {
        ...DEFAULT_AUGMENTATION_CONFIG,
        mosaic: 1.0,
        mixup: 0.1,
        copy_paste: 0.1,
      },
    },
  },
  custom: {
    name: 'Custom',
    description: 'Manually configured settings',
    config: DEFAULT_TRAINING_CONFIG,
  },
};

export const INFERENCE_PRESETS = {
  fast: {
    name: 'Fast Inference',
    description: 'Optimized for speed',
    config: {
      ...DEFAULT_INFERENCE_CONFIG,
      confidence_threshold: 0.5,
      iou_threshold: 0.5,
      max_detections: 100,
      half_precision: true,
    },
  },
  balanced: {
    name: 'Balanced',
    description: 'Good balance between speed and accuracy',
    config: DEFAULT_INFERENCE_CONFIG,
  },
  accurate: {
    name: 'High Accuracy',
    description: 'Optimized for accuracy',
    config: {
      ...DEFAULT_INFERENCE_CONFIG,
      confidence_threshold: 0.1,
      iou_threshold: 0.3,
      max_detections: 1000,
      half_precision: false,
    },
  },
  custom: {
    name: 'Custom',
    description: 'Manually configured settings',
    config: DEFAULT_INFERENCE_CONFIG,
  },
};

// API utilities
export const fetchModels = async (): Promise<ModelInfo[]> => {
  const response = await fetch('/api/v1/models');
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`);
  }
  return response.json();
};

export const fetchModelById = async (modelId: string): Promise<ModelInfo> => {
  const response = await fetch(`/api/v1/models/${modelId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.statusText}`);
  }
  return response.json();
};

export const createModel = async (modelData: Omit<ModelInfo, 'id' | 'created_at' | 'updated_at'>): Promise<ModelInfo> => {
  const response = await fetch('/api/v1/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(modelData),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to create model: ${response.statusText}`);
  }
  
  return response.json();
};

export const updateModel = async (modelId: string, updates: Partial<ModelInfo>): Promise<ModelInfo> => {
  const response = await fetch(`/api/v1/models/${modelId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to update model: ${response.statusText}`);
  }
  
  return response.json();
};

export const deleteModel = async (modelId: string): Promise<void> => {
  const response = await fetch(`/api/v1/models/${modelId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete model: ${response.statusText}`);
  }
};

export const downloadPretrainedModel = async (model: PretrainedModel, onProgress?: (progress: number) => void): Promise<void> => {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = (event.loaded / event.total) * 100;
        onProgress(progress);
      }
    };
    
    xhr.onload = () => {
      if (xhr.status === 200) {
        resolve();
      } else {
        reject(new Error(`Download failed: ${xhr.statusText}`));
      }
    };
    
    xhr.onerror = () => reject(new Error('Download failed'));
    
    xhr.open('POST', '/api/v1/models/download');
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({ model_id: model.id, download_url: model.download_url }));
  });
};

export const validateModel = async (modelId: string, testImages: string[]): Promise<any> => {
  const response = await fetch('/api/v1/models/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, test_images: testImages }),
  });
  
  if (!response.ok) {
    throw new Error(`Model validation failed: ${response.statusText}`);
  }
  
  return response.json();
};

export const exportModel = async (modelId: string, format: string, options: any): Promise<Blob> => {
  const response = await fetch('/api/v1/models/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, format, options }),
  });
  
  if (!response.ok) {
    throw new Error(`Model export failed: ${response.statusText}`);
  }
  
  return response.blob();
};