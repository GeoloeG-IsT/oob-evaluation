// T080: Performance Evaluation Dashboard - Utility Functions

import { EvaluationResult, PerformanceMetrics, ClassMetrics, StatisticalTestResult } from './types';

// Metric formatting utilities
export const formatMetricValue = (value: number | undefined, metricType: string): string => {
  if (value === undefined || value === null) return 'N/A';

  switch (metricType) {
    case 'map':
    case 'map_50':
    case 'map_75':
    case 'accuracy':
    case 'precision':
    case 'recall':
    case 'f1_score':
      return `${(value * 100).toFixed(1)}%`;
    
    case 'inference_time_mean':
    case 'inference_time_std':
      return value < 1 ? `${(value * 1000).toFixed(1)}ms` : `${value.toFixed(3)}s`;
    
    case 'fps':
      return `${Math.round(value)} FPS`;
    
    case 'memory_usage_peak':
    case 'memory_usage_mean':
    case 'model_size':
      return formatBytes(value);
    
    case 'parameters':
      return formatParameters(value);
    
    case 'flops':
      return formatFLOPS(value);
    
    default:
      if (metricType.includes('time')) {
        return formatTime(value);
      } else if (metricType.includes('size') || metricType.includes('memory')) {
        return formatBytes(value);
      } else {
        return value.toFixed(4);
      }
  }
};

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

export const formatParameters = (params: number): string => {
  if (params >= 1e9) {
    return `${(params / 1e9).toFixed(1)}B`;
  } else if (params >= 1e6) {
    return `${(params / 1e6).toFixed(1)}M`;
  } else if (params >= 1e3) {
    return `${(params / 1e3).toFixed(1)}K`;
  } else {
    return params.toString();
  }
};

export const formatFLOPS = (flops: number): string => {
  if (flops >= 1e12) {
    return `${(flops / 1e12).toFixed(1)}T`;
  } else if (flops >= 1e9) {
    return `${(flops / 1e9).toFixed(1)}G`;
  } else if (flops >= 1e6) {
    return `${(flops / 1e6).toFixed(1)}M`;
  } else if (flops >= 1e3) {
    return `${(flops / 1e3).toFixed(1)}K`;
  } else {
    return flops.toString();
  }
};

export const formatTime = (seconds: number): string => {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(1)}ms`;
  } else if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs.toFixed(0)}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
};

// Metric analysis utilities
export const getMetricCategory = (metricName: string): string => {
  if (['map', 'map_50', 'map_75', 'accuracy', 'precision', 'recall', 'f1_score'].includes(metricName)) {
    return 'accuracy';
  } else if (['inference_time_mean', 'inference_time_std', 'fps'].includes(metricName)) {
    return 'speed';
  } else if (['memory_usage_peak', 'memory_usage_mean', 'model_size'].includes(metricName)) {
    return 'efficiency';
  } else if (['parameters', 'flops'].includes(metricName)) {
    return 'complexity';
  } else {
    return 'other';
  }
};

export const isHigherBetter = (metricName: string): boolean => {
  const higherIsBetter = [
    'map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large',
    'accuracy', 'precision', 'recall', 'f1_score',
    'precision_macro', 'recall_macro', 'f1_macro',
    'precision_weighted', 'recall_weighted', 'f1_weighted',
    'fps'
  ];
  
  const lowerIsBetter = [
    'inference_time_mean', 'inference_time_std',
    'memory_usage_peak', 'memory_usage_mean',
    'model_size', 'parameters', 'flops'
  ];
  
  if (higherIsBetter.includes(metricName)) {
    return true;
  } else if (lowerIsBetter.includes(metricName)) {
    return false;
  } else {
    // Default assumption for unknown metrics
    return !metricName.includes('time') && !metricName.includes('size') && !metricName.includes('memory');
  }
};

export const calculateRelativeImprovement = (
  newValue: number,
  baselineValue: number,
  metricName: string
): number => {
  if (baselineValue === 0) return 0;
  
  const improvement = (newValue - baselineValue) / baselineValue;
  return isHigherBetter(metricName) ? improvement : -improvement;
};

export const rankEvaluations = (
  evaluations: EvaluationResult[],
  metric: string
): EvaluationResult[] => {
  const completedEvaluations = evaluations.filter(e => e.status === 'completed');
  
  return completedEvaluations.sort((a, b) => {
    const aValue = (a.metrics as any)[metric] || 0;
    const bValue = (b.metrics as any)[metric] || 0;
    
    return isHigherBetter(metric) ? bValue - aValue : aValue - bValue;
  });
};

// Statistical utilities
export const calculateMean = (values: number[]): number => {
  return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0;
};

export const calculateStandardDeviation = (values: number[]): number => {
  const mean = calculateMean(values);
  const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
  const variance = calculateMean(squaredDiffs);
  return Math.sqrt(variance);
};

export const calculateConfidenceInterval = (
  values: number[],
  confidence: number = 0.95
): [number, number] => {
  const mean = calculateMean(values);
  const std = calculateStandardDeviation(values);
  const n = values.length;
  
  // Using t-distribution critical value (approximation for large n)
  const tValue = confidence === 0.95 ? 1.96 : confidence === 0.99 ? 2.576 : 1.645;
  const margin = (tValue * std) / Math.sqrt(n);
  
  return [mean - margin, mean + margin];
};

export const performTTest = (
  group1: number[],
  group2: number[],
  significanceLevel: number = 0.05
): StatisticalTestResult => {
  const mean1 = calculateMean(group1);
  const mean2 = calculateMean(group2);
  const n1 = group1.length;
  const n2 = group2.length;
  
  const var1 = group1.reduce((sum, val) => sum + Math.pow(val - mean1, 2), 0) / (n1 - 1);
  const var2 = group2.reduce((sum, val) => sum + Math.pow(val - mean2, 2), 0) / (n2 - 1);
  
  const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
  const standardError = Math.sqrt(pooledVar * (1/n1 + 1/n2));
  
  const tStatistic = (mean1 - mean2) / standardError;
  
  // Simplified p-value calculation (approximation)
  const degreesOfFreedom = n1 + n2 - 2;
  const pValue = 2 * (1 - normalCDF(Math.abs(tStatistic)));
  
  const significant = pValue < significanceLevel;
  const effectSize = Math.abs(mean1 - mean2) / Math.sqrt(pooledVar);
  
  return {
    test_type: 'ttest',
    statistic: tStatistic,
    p_value: pValue,
    significant,
    confidence_level: 1 - significanceLevel,
    effect_size: effectSize,
    interpretation: significant 
      ? `There is a statistically significant difference between the groups (p=${pValue.toFixed(4)}).`
      : `There is no statistically significant difference between the groups (p=${pValue.toFixed(4)}).`,
    recommendations: significant
      ? ['The observed difference is likely not due to random chance.', 'Consider the practical significance of this difference.']
      : ['The observed difference could be due to random variation.', 'Consider increasing sample size or checking measurement precision.']
  };
};

// Helper function for normal CDF approximation
const normalCDF = (x: number): number => {
  return (1.0 + erf(x / Math.sqrt(2.0))) / 2.0;
};

const erf = (x: number): number => {
  // Approximation of the error function
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;
  
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  
  return sign * y;
};

// Class metrics utilities
export const calculateClassMetricsSummary = (classMetrics: ClassMetrics[]) => {
  const totalSupport = classMetrics.reduce((sum, cm) => sum + cm.support, 0);
  
  const macroAvg = {
    precision: calculateMean(classMetrics.map(cm => cm.precision)),
    recall: calculateMean(classMetrics.map(cm => cm.recall)),
    f1_score: calculateMean(classMetrics.map(cm => cm.f1_score)),
  };
  
  const weightedAvg = {
    precision: classMetrics.reduce((sum, cm) => sum + cm.precision * cm.support, 0) / totalSupport,
    recall: classMetrics.reduce((sum, cm) => sum + cm.recall * cm.support, 0) / totalSupport,
    f1_score: classMetrics.reduce((sum, cm) => sum + cm.f1_score * cm.support, 0) / totalSupport,
  };
  
  return { macroAvg, weightedAvg, totalSupport };
};

export const findBestAndWorstClasses = (
  classMetrics: ClassMetrics[],
  metric: keyof ClassMetrics
): { best: ClassMetrics; worst: ClassMetrics } | null => {
  if (classMetrics.length === 0) return null;
  
  const values = classMetrics.map(cm => cm[metric] as number);
  const maxValue = Math.max(...values);
  const minValue = Math.min(...values);
  
  const best = classMetrics.find(cm => cm[metric] === maxValue)!;
  const worst = classMetrics.find(cm => cm[metric] === minValue)!;
  
  return { best, worst };
};

// Data export utilities
export const exportToCSV = (data: any[], filename: string): void => {
  if (data.length === 0) return;
  
  const headers = Object.keys(data[0]).join(',');
  const rows = data.map(row => 
    Object.values(row).map(value => 
      typeof value === 'string' ? `"${value}"` : value
    ).join(',')
  );
  
  const csv = [headers, ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  
  URL.revokeObjectURL(url);
};

export const exportToJSON = (data: any, filename: string): void => {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  
  URL.revokeObjectURL(url);
};

// API utilities
export const fetchEvaluations = async (): Promise<EvaluationResult[]> => {
  const response = await fetch('/api/v1/evaluations');
  if (!response.ok) {
    throw new Error(`Failed to fetch evaluations: ${response.statusText}`);
  }
  return response.json();
};

export const createEvaluation = async (
  modelId: string,
  datasetId: string,
  config: any
): Promise<EvaluationResult> => {
  const response = await fetch('/api/v1/evaluations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_id: modelId,
      dataset_id: datasetId,
      config
    })
  });
  
  if (!response.ok) {
    throw new Error(`Failed to create evaluation: ${response.statusText}`);
  }
  
  return response.json();
};

export const deleteEvaluation = async (evaluationId: string): Promise<void> => {
  const response = await fetch(`/api/v1/evaluations/${evaluationId}`, {
    method: 'DELETE'
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete evaluation: ${response.statusText}`);
  }
};

export const exportEvaluation = async (
  evaluationId: string,
  format: string
): Promise<Blob> => {
  const response = await fetch(`/api/v1/evaluations/${evaluationId}/export?format=${format}`);
  
  if (!response.ok) {
    throw new Error(`Failed to export evaluation: ${response.statusText}`);
  }
  
  return response.blob();
};