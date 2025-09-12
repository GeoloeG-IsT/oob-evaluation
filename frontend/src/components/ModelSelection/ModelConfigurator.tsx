'use client';

import React, { useState } from 'react';
import { ModelConfiguratorProps, TrainingConfig, InferenceConfig } from './types';
import { TrainingConfigForm } from './TrainingConfigForm';
import { InferenceConfigForm } from './InferenceConfigForm';
import { validateTrainingConfig, validateInferenceConfig, TRAINING_PRESETS, INFERENCE_PRESETS } from './utils';

export const ModelConfigurator: React.FC<ModelConfiguratorProps> = ({
  model,
  config,
  configType,
  onConfigChange,
  onSave,
  onReset,
  isReadOnly = false,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'basic' | 'advanced'>('basic');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const handleConfigChange = (newConfig: TrainingConfig | InferenceConfig) => {
    // Validate configuration
    let validation;
    if (configType === 'training') {
      validation = validateTrainingConfig(newConfig as TrainingConfig);
    } else {
      validation = validateInferenceConfig(newConfig as InferenceConfig);
    }
    
    setValidationErrors(validation.errors);
    onConfigChange(newConfig);
  };

  const handlePresetSelect = (presetKey: string) => {
    const presets = configType === 'training' ? TRAINING_PRESETS : INFERENCE_PRESETS;
    const preset = presets[presetKey as keyof typeof presets];
    if (preset) {
      handleConfigChange(preset.config);
    }
  };

  const handleSave = () => {
    if (validationErrors.length === 0) {
      onSave?.();
    }
  };

  const presets = configType === 'training' ? TRAINING_PRESETS : INFERENCE_PRESETS;

  return (
    <div className={`model-configurator bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {configType === 'training' ? 'Training' : 'Inference'} Configuration
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Configure {model.name} for {configType === 'training' ? 'training' : 'inference'}
            </p>
          </div>
          
          {!isReadOnly && (
            <div className="flex space-x-2">
              {onReset && (
                <button
                  onClick={onReset}
                  className="px-3 py-2 text-sm bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-500"
                >
                  Reset
                </button>
              )}
              {onSave && (
                <button
                  onClick={handleSave}
                  disabled={validationErrors.length > 0}
                  className={`
                    px-4 py-2 text-sm font-medium rounded transition-colors
                    ${validationErrors.length > 0
                      ? 'bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-600 cursor-not-allowed'
                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                    }
                  `}
                >
                  Save Configuration
                </button>
              )}
            </div>
          )}
        </div>

        {/* Validation Errors */}
        {validationErrors.length > 0 && (
          <div className="mb-4 p-3 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded">
            <h4 className="font-medium text-red-800 dark:text-red-200 mb-2">
              Configuration Errors:
            </h4>
            <ul className="text-sm text-red-700 dark:text-red-300 space-y-1">
              {validationErrors.map((error, index) => (
                <li key={index}>• {error}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Presets */}
        <div className="space-y-3">
          <h4 className="font-medium text-gray-900 dark:text-gray-100">Quick Presets</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2">
            {Object.entries(presets).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => handlePresetSelect(key)}
                disabled={isReadOnly}
                className={`
                  p-3 text-left border rounded-lg transition-colors
                  ${isReadOnly
                    ? 'border-gray-200 dark:border-gray-700 text-gray-400 dark:text-gray-600 cursor-not-allowed'
                    : 'border-gray-200 dark:border-gray-600 hover:border-blue-300 dark:hover:border-blue-700 hover:bg-blue-50 dark:hover:bg-blue-950'
                  }
                `}
              >
                <div className="font-medium text-gray-900 dark:text-gray-100 text-sm">
                  {preset.name}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  {preset.description}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex">
          <button
            onClick={() => setActiveTab('basic')}
            className={`
              px-6 py-3 text-sm font-medium border-b-2 transition-colors
              ${activeTab === 'basic'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
              }
            `}
          >
            Basic Settings
          </button>
          <button
            onClick={() => setActiveTab('advanced')}
            className={`
              px-6 py-3 text-sm font-medium border-b-2 transition-colors
              ${activeTab === 'advanced'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
              }
            `}
          >
            Advanced Settings
          </button>
        </nav>
      </div>

      {/* Configuration Form */}
      <div className="p-6">
        {configType === 'training' ? (
          <TrainingConfigForm
            config={config as TrainingConfig}
            onChange={handleConfigChange}
            modelType={model.type}
            isReadOnly={isReadOnly}
            activeTab={activeTab}
          />
        ) : (
          <InferenceConfigForm
            config={config as InferenceConfig}
            onChange={handleConfigChange}
            modelType={model.type}
            isReadOnly={isReadOnly}
            activeTab={activeTab}
          />
        )}
      </div>

      {/* Model-specific recommendations */}
      <div className="p-6 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-750">
        <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
          Recommendations for {model.architecture.toUpperCase()}
        </h4>
        
        <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
          {model.architecture.startsWith('yolo') && configType === 'training' && (
            <>
              <p>• Use mosaic augmentation for better object detection performance</p>
              <p>• Start with batch size 16-32 for optimal GPU memory usage</p>
              <p>• Enable mixed precision for faster training on modern GPUs</p>
            </>
          )}
          
          {model.architecture === 'rtdetr' && configType === 'training' && (
            <>
              <p>• RT-DETR works best with larger batch sizes (32+) when possible</p>
              <p>• Use AdamW optimizer for transformer-based architecture</p>
              <p>• Consider longer warmup periods (5-10 epochs)</p>
            </>
          )}
          
          {model.architecture === 'sam2' && configType === 'training' && (
            <>
              <p>• SAM2 requires large memory - reduce batch size if needed</p>
              <p>• Focus on high-quality annotations for segmentation</p>
              <p>• Use specialized loss functions for segmentation tasks</p>
            </>
          )}
          
          {configType === 'inference' && (
            <>
              <p>• Lower confidence threshold for better recall, higher for precision</p>
              <p>• Adjust IoU threshold based on object density in your images</p>
              <p>• Enable half precision for 2x inference speed on compatible hardware</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};