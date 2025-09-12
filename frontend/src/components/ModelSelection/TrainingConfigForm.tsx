'use client';

import React from 'react';
import { TrainingConfig, AugmentationConfig } from './types';

interface TrainingConfigFormProps {
  config: TrainingConfig;
  onChange: (config: TrainingConfig) => void;
  modelType: 'detection' | 'segmentation';
  isReadOnly?: boolean;
  activeTab?: 'basic' | 'advanced';
}

export const TrainingConfigForm: React.FC<TrainingConfigFormProps> = ({
  config,
  onChange,
  modelType,
  isReadOnly = false,
  activeTab = 'basic',
}) => {
  const handleConfigChange = (updates: Partial<TrainingConfig>) => {
    onChange({ ...config, ...updates });
  };

  const handleAugmentationChange = (updates: Partial<AugmentationConfig>) => {
    handleConfigChange({
      augmentation: { ...config.augmentation, ...updates },
    });
  };

  const handleEarlyStoppingChange = (updates: Partial<TrainingConfig['early_stopping']>) => {
    handleConfigChange({
      early_stopping: { ...config.early_stopping, ...updates },
    });
  };

  if (activeTab === 'basic') {
    return (
      <div className="space-y-6">
        {/* Training Parameters */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Training Parameters
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Epochs
              </label>
              <input
                type="number"
                value={config.epochs}
                onChange={(e) => handleConfigChange({ epochs: parseInt(e.target.value) || 100 })}
                disabled={isReadOnly}
                min="1"
                max="10000"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Number of complete passes through the dataset
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
                {[1, 2, 4, 8, 16, 32, 64, 128].map(size => (
                  <option key={size} value={size}>{size}</option>
                ))}
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Number of samples processed before model update
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                value={config.learning_rate}
                onChange={(e) => handleConfigChange({ learning_rate: parseFloat(e.target.value) || 0.01 })}
                disabled={isReadOnly}
                step="0.001"
                min="0.0001"
                max="1"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Step size for gradient descent
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Optimizer
              </label>
              <select
                value={config.optimizer}
                onChange={(e) => handleConfigChange({ optimizer: e.target.value as TrainingConfig['optimizer'] })}
                disabled={isReadOnly}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              >
                <option value="sgd">SGD</option>
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Optimization algorithm
              </p>
            </div>
          </div>
        </div>

        {/* Learning Rate Scheduler */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Learning Rate Schedule
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Scheduler Type
              </label>
              <select
                value={config.lr_scheduler}
                onChange={(e) => handleConfigChange({ lr_scheduler: e.target.value as TrainingConfig['lr_scheduler'] })}
                disabled={isReadOnly}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              >
                <option value="cosine">Cosine Annealing</option>
                <option value="linear">Linear</option>
                <option value="step">Step Decay</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Warmup Epochs
              </label>
              <input
                type="number"
                value={config.warmup_epochs}
                onChange={(e) => handleConfigChange({ warmup_epochs: parseInt(e.target.value) || 3 })}
                disabled={isReadOnly}
                min="0"
                max={Math.floor(config.epochs / 2)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              />
            </div>
          </div>
        </div>

        {/* Device and Performance */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Device & Performance
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Device
              </label>
              <select
                value={config.device}
                onChange={(e) => handleConfigChange({ device: e.target.value as TrainingConfig['device'] })}
                disabled={isReadOnly}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              >
                <option value="auto">Auto (Detect Best)</option>
                <option value="cuda">GPU (CUDA)</option>
                <option value="cpu">CPU Only</option>
              </select>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="mixed_precision"
                checked={config.mixed_precision}
                onChange={(e) => handleConfigChange({ mixed_precision: e.target.checked })}
                disabled={isReadOnly}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="mixed_precision" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                Mixed Precision Training
              </label>
              <div className="ml-2 text-xs text-gray-500 dark:text-gray-400">
                (Faster training with modern GPUs)
              </div>
            </div>
          </div>
        </div>

        {/* Basic Augmentation */}
        <div>
          <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
            Data Augmentation
          </h4>
          <div className="space-y-3">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="augmentation_enabled"
                checked={config.augmentation.enabled}
                onChange={(e) => handleAugmentationChange({ enabled: e.target.checked })}
                disabled={isReadOnly}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="augmentation_enabled" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                Enable Data Augmentation
              </label>
            </div>

            {config.augmentation.enabled && (
              <div className="ml-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Horizontal Flip Probability
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.augmentation.fliplr}
                    onChange={(e) => handleAugmentationChange({ fliplr: parseFloat(e.target.value) })}
                    disabled={isReadOnly}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {(config.augmentation.fliplr * 100).toFixed(0)}%
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Mosaic Probability
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.augmentation.mosaic}
                    onChange={(e) => handleAugmentationChange({ mosaic: parseFloat(e.target.value) })}
                    disabled={isReadOnly}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {(config.augmentation.mosaic * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Advanced Tab
  return (
    <div className="space-y-6">
      {/* Advanced Training Parameters */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
          Advanced Parameters
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Weight Decay
            </label>
            <input
              type="number"
              value={config.weight_decay}
              onChange={(e) => handleConfigChange({ weight_decay: parseFloat(e.target.value) || 0.0005 })}
              disabled={isReadOnly}
              step="0.0001"
              min="0"
              max="0.1"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Momentum (SGD only)
            </label>
            <input
              type="number"
              value={config.momentum}
              onChange={(e) => handleConfigChange({ momentum: parseFloat(e.target.value) || 0.937 })}
              disabled={isReadOnly || config.optimizer !== 'sgd'}
              step="0.01"
              min="0"
              max="1"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Gradient Clipping
            </label>
            <input
              type="number"
              value={config.gradient_clip}
              onChange={(e) => handleConfigChange({ gradient_clip: parseFloat(e.target.value) || 10.0 })}
              disabled={isReadOnly}
              step="0.1"
              min="0"
              max="100"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Validation Period
            </label>
            <input
              type="number"
              value={config.val_period}
              onChange={(e) => handleConfigChange({ val_period: parseInt(e.target.value) || 1 })}
              disabled={isReadOnly}
              min="1"
              max="50"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Validate every N epochs
            </p>
          </div>
        </div>
      </div>

      {/* Early Stopping */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
          Early Stopping
        </h4>
        <div className="space-y-3">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="early_stopping_enabled"
              checked={config.early_stopping.enabled}
              onChange={(e) => handleEarlyStoppingChange({ enabled: e.target.checked })}
              disabled={isReadOnly}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
            <label htmlFor="early_stopping_enabled" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
              Enable Early Stopping
            </label>
          </div>

          {config.early_stopping.enabled && (
            <div className="ml-6 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Patience (epochs)
                </label>
                <input
                  type="number"
                  value={config.early_stopping.patience}
                  onChange={(e) => handleEarlyStoppingChange({ patience: parseInt(e.target.value) || 50 })}
                  disabled={isReadOnly}
                  min="1"
                  max="200"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Minimum Delta
                </label>
                <input
                  type="number"
                  value={config.early_stopping.min_delta}
                  onChange={(e) => handleEarlyStoppingChange({ min_delta: parseFloat(e.target.value) || 0.001 })}
                  disabled={isReadOnly}
                  step="0.001"
                  min="0"
                  max="1"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Advanced Augmentation */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
          Advanced Augmentation
        </h4>
        {config.augmentation.enabled && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Color augmentations */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                HSV Hue
              </label>
              <input
                type="range"
                min="0"
                max="0.1"
                step="0.005"
                value={config.augmentation.hsv_h}
                onChange={(e) => handleAugmentationChange({ hsv_h: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full"
              />
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {config.augmentation.hsv_h.toFixed(3)}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                HSV Saturation
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.augmentation.hsv_s}
                onChange={(e) => handleAugmentationChange({ hsv_s: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full"
              />
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {config.augmentation.hsv_s.toFixed(1)}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                HSV Value
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.augmentation.hsv_v}
                onChange={(e) => handleAugmentationChange({ hsv_v: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full"
              />
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {config.augmentation.hsv_v.toFixed(1)}
              </div>
            </div>

            {/* Geometric augmentations */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Rotation Degrees
              </label>
              <input
                type="range"
                min="0"
                max="45"
                step="5"
                value={config.augmentation.degrees}
                onChange={(e) => handleAugmentationChange({ degrees: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full"
              />
              <div className="text-xs text-gray-500 dark:text-gray-400">
                ±{config.augmentation.degrees}°
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Translation
              </label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={config.augmentation.translate}
                onChange={(e) => handleAugmentationChange({ translate: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full"
              />
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {(config.augmentation.translate * 100).toFixed(0)}%
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Scale
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.augmentation.scale}
                onChange={(e) => handleAugmentationChange({ scale: parseFloat(e.target.value) })}
                disabled={isReadOnly}
                className="w-full"
              />
              <div className="text-xs text-gray-500 dark:text-gray-400">
                ±{(config.augmentation.scale * 100).toFixed(0)}%
              </div>
            </div>

            {/* Advanced augmentations for detection */}
            {modelType === 'detection' && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Mixup
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.augmentation.mixup}
                    onChange={(e) => handleAugmentationChange({ mixup: parseFloat(e.target.value) })}
                    disabled={isReadOnly}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {(config.augmentation.mixup * 100).toFixed(0)}%
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Copy-Paste
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.augmentation.copy_paste}
                    onChange={(e) => handleAugmentationChange({ copy_paste: parseFloat(e.target.value) })}
                    disabled={isReadOnly}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {(config.augmentation.copy_paste * 100).toFixed(0)}%
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};