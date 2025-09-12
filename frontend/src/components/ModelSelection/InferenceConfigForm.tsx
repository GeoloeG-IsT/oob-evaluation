'use client';

import React from 'react';
import { InferenceConfig } from './types';

interface InferenceConfigFormProps {
  config: InferenceConfig;
  onChange: (config: InferenceConfig) => void;
  modelType: 'detection' | 'segmentation';
  isReadOnly?: boolean;
  activeTab?: 'basic' | 'advanced';
}

export const InferenceConfigForm: React.FC<InferenceConfigFormProps> = ({
  config,
  onChange,
  modelType,
  isReadOnly = false,
  activeTab = 'basic',
}) => {
  const handleConfigChange = (updates: Partial<InferenceConfig>) => {
    onChange({ ...config, ...updates });
  };

  if (activeTab === 'basic') {
    return (
      <div className="space-y-6">
        {/* Detection Thresholds */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Detection Thresholds
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Confidence Threshold
              </label>
              <input
                type="range"
                min="0.01"
                max="0.99"
                step="0.01"
                value={config.confidence_threshold}
                onChange={(e) => handleConfigChange({ confidence_threshold: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full mb-2"
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>Low (0.01)</span>
                <span className="font-medium">{(config.confidence_threshold * 100).toFixed(0)}%</span>
                <span>High (0.99)</span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Minimum confidence for detections to be considered valid
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                IoU Threshold (NMS)
              </label>
              <input
                type="range"
                min="0.01"
                max="0.99"
                step="0.01"
                value={config.iou_threshold}
                onChange={(e) => handleConfigChange({ iou_threshold: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full mb-2"
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>Strict (0.01)</span>
                <span className="font-medium">{(config.iou_threshold * 100).toFixed(0)}%</span>
                <span>Relaxed (0.99)</span>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Threshold for Non-Maximum Suppression (duplicate removal)
              </p>
            </div>
          </div>

          {/* Threshold Guidance */}
          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
            <h5 className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2">
              Threshold Guidelines
            </h5>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-blue-700 dark:text-blue-300">
              <div>
                <div className="font-medium mb-1">Confidence Threshold:</div>
                <ul className="space-y-1">
                  <li>• High (0.7-0.9): Fewer false positives, may miss objects</li>
                  <li>• Medium (0.3-0.6): Balanced precision and recall</li>
                  <li>• Low (0.1-0.3): More detections, more false positives</li>
                </ul>
              </div>
              <div>
                <div className="font-medium mb-1">IoU Threshold:</div>
                <ul className="space-y-1">
                  <li>• High (0.6-0.8): Keep similar overlapping boxes</li>
                  <li>• Medium (0.4-0.6): Standard duplicate removal</li>
                  <li>• Low (0.1-0.4): Aggressive duplicate removal</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Output Configuration */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Output Configuration
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Maximum Detections
              </label>
              <select
                value={config.max_detections}
                onChange={(e) => handleConfigChange({ max_detections: parseInt(e.target.value) })}
                disabled={isReadOnly}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              >
                <option value={100}>100 (Fast)</option>
                <option value={300}>300 (Standard)</option>
                <option value={1000}>1000 (Comprehensive)</option>
                <option value={5000}>5000 (Maximum)</option>
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Maximum number of objects to detect per image
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Batch Size
              </label>
              <select
                value={config.batch_size}
                onChange={(e) => handleConfigChange({ batch_size: parseInt(e.target.value) })}
                disabled={isReadOnly}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              >
                {[1, 2, 4, 8, 16, 32].map(size => (
                  <option key={size} value={size}>
                    {size} {size === 1 ? '(Single)' : size <= 4 ? '(Small batch)' : size <= 16 ? '(Medium batch)' : '(Large batch)'}
                  </option>
                ))}
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Number of images to process simultaneously
              </p>
            </div>
          </div>
        </div>

        {/* Performance Settings */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Performance Settings
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Device
              </label>
              <select
                value={config.device}
                onChange={(e) => handleConfigChange({ device: e.target.value as InferenceConfig['device'] })}
                disabled={isReadOnly}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              >
                <option value="auto">Auto (Detect Best)</option>
                <option value="cuda">GPU (CUDA)</option>
                <option value="cpu">CPU Only</option>
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Processing device selection
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="half_precision"
                  checked={config.half_precision}
                  onChange={(e) => handleConfigChange({ half_precision: e.target.checked })}
                  disabled={isReadOnly}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="half_precision" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                  Half Precision (FP16)
                </label>
              </div>
            </div>
            <div className="col-span-2">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Half precision can provide 2x speed improvement on compatible GPUs with minimal accuracy loss
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Advanced Tab
  return (
    <div className="space-y-6">
      {/* Advanced NMS Settings */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
          Advanced NMS Settings
        </h4>
        <div className="space-y-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="agnostic_nms"
                checked={config.agnostic_nms}
                onChange={(e) => handleConfigChange({ agnostic_nms: e.target.checked })}
                disabled={isReadOnly}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="agnostic_nms" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                Class-Agnostic NMS
              </label>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="multi_label"
                checked={config.multi_label}
                onChange={(e) => handleConfigChange({ multi_label: e.target.checked })}
                disabled={isReadOnly}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="multi_label" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                Multi-Label Detection
              </label>
            </div>
          </div>
          
          <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
            <p>• Class-Agnostic NMS: Apply NMS across all classes (useful for overlapping objects of different types)</p>
            <p>• Multi-Label: Allow objects to belong to multiple classes simultaneously</p>
          </div>
        </div>
      </div>

      {/* Class Filtering */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
          Class Filtering
        </h4>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Specific Classes Only (Optional)
          </label>
          <textarea
            value={config.classes?.join(', ') || ''}
            onChange={(e) => {
              const classesStr = e.target.value.trim();
              const classes = classesStr ? classesStr.split(',').map(c => parseInt(c.trim())).filter(c => !isNaN(c)) : undefined;
              handleConfigChange({ classes });
            }}
            disabled={isReadOnly}
            placeholder="e.g., 0, 1, 2 (leave empty for all classes)"
            rows={2}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Comma-separated list of class indices to detect. Leave empty to detect all classes.
          </p>
        </div>
      </div>

      {/* Model-specific optimizations */}
      {modelType === 'detection' && (
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Detection Optimizations
          </h4>
          <div className="space-y-3">
            <div className="p-3 bg-gray-50 dark:bg-gray-750 rounded">
              <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">YOLO-specific</h5>
              <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Use confidence threshold 0.25-0.5 for best results</li>
                <li>• IoU threshold 0.45 works well for most scenarios</li>
                <li>• Enable half precision for 2x speed on V100/A100 GPUs</li>
              </ul>
            </div>
            
            <div className="p-3 bg-gray-50 dark:bg-gray-750 rounded">
              <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">RT-DETR-specific</h5>
              <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Can use lower confidence thresholds (0.1-0.3)</li>
                <li>• Built-in NMS may require higher IoU thresholds</li>
                <li>• Larger batch sizes improve efficiency</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {modelType === 'segmentation' && (
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Segmentation Settings
          </h4>
          <div className="space-y-3">
            <div className="p-3 bg-gray-50 dark:bg-gray-750 rounded">
              <h5 className="font-medium text-gray-900 dark:text-gray-100 mb-2">SAM2-specific</h5>
              <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Lower confidence thresholds (0.1-0.3) for fine details</li>
                <li>• Batch size 1-2 recommended for memory efficiency</li>
                <li>• Half precision may affect segmentation quality</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Performance Monitoring */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
          Performance Monitoring
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded">
            <h5 className="font-medium text-blue-800 dark:text-blue-200 mb-2">Speed Optimization</h5>
            <ul className="text-xs text-blue-700 dark:text-blue-300 space-y-1">
              <li>✓ Use GPU when available</li>
              <li>✓ Enable half precision</li>
              <li>✓ Increase batch size</li>
              <li>✓ Higher confidence threshold</li>
              <li>✓ Lower max detections</li>
            </ul>
          </div>
          
          <div className="p-3 bg-green-50 dark:bg-green-950 rounded">
            <h5 className="font-medium text-green-800 dark:text-green-200 mb-2">Accuracy Optimization</h5>
            <ul className="text-xs text-green-700 dark:text-green-300 space-y-1">
              <li>✓ Lower confidence threshold</li>
              <li>✓ Disable half precision</li>
              <li>✓ Higher max detections</li>
              <li>✓ Fine-tune IoU threshold</li>
              <li>✓ Use full precision mode</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};