'use client';

import React, { useState } from 'react';
import { AssistedAnnotationProps, AssistanceModel } from './types';
import { getAssistedAnnotationsAPI } from './utils';

export const AssistedAnnotation: React.FC<AssistedAnnotationProps> = ({
  imageUrl,
  onAnnotationsGenerated,
  onError,
  models,
  selectedModelId,
  onModelSelect,
  className = '',
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');

  const selectedModel = models.find(m => m.id === selectedModelId) || models[0];

  const handleGenerateAnnotations = async () => {
    if (!selectedModel || !imageUrl) return;

    setIsGenerating(true);
    setProgress(0);
    setProgressMessage('Initializing model...');

    try {
      // Simulate progress updates
      const progressSteps = [
        { progress: 20, message: 'Loading image...' },
        { progress: 40, message: 'Running inference...' },
        { progress: 60, message: 'Processing detections...' },
        { progress: 80, message: 'Creating annotations...' },
        { progress: 100, message: 'Complete!' },
      ];

      for (const step of progressSteps) {
        await new Promise(resolve => setTimeout(resolve, 500));
        setProgress(step.progress);
        setProgressMessage(step.message);
      }

      // Make API call
      const result = await getAssistedAnnotationsAPI(imageUrl, selectedModel.name);
      
      if (result.annotations && result.annotations.length > 0) {
        onAnnotationsGenerated?.(result.annotations);
      } else {
        onError?.('No objects detected in the image.');
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate annotations';
      onError?.(errorMessage);
    } finally {
      setIsGenerating(false);
      setProgress(0);
      setProgressMessage('');
    }
  };

  const getModelTypeIcon = (type: string) => {
    switch (type) {
      case 'detection':
        return '▢';
      case 'segmentation':
        return '⬟';
      case 'sam':
        return '✂';
      default:
        return '🤖';
    }
  };

  const getModelTypeColor = (type: string) => {
    switch (type) {
      case 'detection':
        return 'text-blue-600 dark:text-blue-400';
      case 'segmentation':
        return 'text-green-600 dark:text-green-400';
      case 'sam':
        return 'text-purple-600 dark:text-purple-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  return (
    <div className={`assisted-annotation bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            AI-Assisted Annotation
          </h3>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            {models.length} models available
          </div>
        </div>
      </div>

      {/* Model Selection */}
      <div className="p-4 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Select AI Model
          </label>
          <div className="space-y-2">
            {models.map((model) => (
              <div
                key={model.id}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedModelId === model.id
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                    : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                }`}
                onClick={() => onModelSelect?.(model.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`text-xl ${getModelTypeColor(model.type)}`}>
                      {getModelTypeIcon(model.type)}
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-gray-100">
                        {model.name}
                      </h4>
                      {model.description && (
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {model.description}
                        </p>
                      )}
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className={`text-sm font-medium capitalize ${getModelTypeColor(model.type)}`}>
                      {model.type}
                    </div>
                    {model.version && (
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        v{model.version}
                      </div>
                    )}
                  </div>
                </div>

                {/* Capabilities */}
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(model.capabilities)
                    .filter(([_, enabled]) => enabled)
                    .map(([capability]) => (
                      <span
                        key={capability}
                        className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                      >
                        {capability}
                      </span>
                    ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Generate Button */}
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {selectedModel && (
              <div>
                Selected: <span className="font-medium">{selectedModel.name}</span>
                {selectedModel.capabilities.interactive && (
                  <span className="ml-2 px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200 rounded">
                    Interactive
                  </span>
                )}
              </div>
            )}
          </div>
          
          <button
            onClick={handleGenerateAnnotations}
            disabled={isGenerating || !selectedModel || !imageUrl}
            className={`
              px-4 py-2 rounded-lg font-medium transition-colors
              ${isGenerating || !selectedModel || !imageUrl
                ? 'bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-600 cursor-not-allowed'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
              }
            `}
          >
            {isGenerating ? 'Generating...' : 'Generate Annotations'}
          </button>
        </div>

        {/* Progress */}
        {isGenerating && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">{progressMessage}</span>
              <span className="text-gray-500 dark:text-gray-500">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Information Panel */}
      {selectedModel && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-750">
          <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
            Model Information
          </h4>
          
          <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="font-medium">Type:</span>{' '}
                <span className="capitalize">{selectedModel.type}</span>
              </div>
              {selectedModel.version && (
                <div>
                  <span className="font-medium">Version:</span> {selectedModel.version}
                </div>
              )}
            </div>

            {selectedModel.supported_formats && selectedModel.supported_formats.length > 0 && (
              <div>
                <span className="font-medium">Supported formats:</span>{' '}
                {selectedModel.supported_formats.join(', ')}
              </div>
            )}

            {/* Usage Instructions */}
            <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-950 rounded text-blue-800 dark:text-blue-200">
              <h5 className="font-medium mb-1">How to use:</h5>
              <ul className="text-xs space-y-1">
                {selectedModel.type === 'detection' && (
                  <>
                    <li>• This model will detect objects and create bounding boxes</li>
                    <li>• Review and adjust the generated annotations as needed</li>
                  </>
                )}
                {selectedModel.type === 'segmentation' && (
                  <>
                    <li>• This model will segment objects and create polygon annotations</li>
                    <li>• Fine-tune the polygon boundaries if necessary</li>
                  </>
                )}
                {selectedModel.type === 'sam' && (
                  <>
                    <li>• This is an interactive segmentation model</li>
                    <li>• Click on objects to generate precise segmentation masks</li>
                  </>
                )}
                <li>• Generated annotations will have confidence scores</li>
                <li>• Low-confidence annotations should be reviewed carefully</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};