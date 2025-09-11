'use client';

import React, { useState, useMemo } from 'react';
import { DatasetSplitManagerProps, UploadedImage } from './types';
import { calculateSplitDistribution } from './utils';

export const DatasetSplitManager: React.FC<DatasetSplitManagerProps> = ({
  images,
  onSplitChange,
  splitDistribution,
  className = '',
}) => {
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [bulkSplit, setBulkSplit] = useState<'train' | 'val' | 'test'>('train');

  // Calculate current distribution
  const currentDistribution = useMemo(() => {
    return splitDistribution || calculateSplitDistribution(images);
  }, [images, splitDistribution]);

  // Calculate percentages
  const percentages = useMemo(() => {
    const total = currentDistribution.total;
    if (total === 0) return { train: 0, val: 0, test: 0 };
    
    return {
      train: Math.round((currentDistribution.train / total) * 100),
      val: Math.round((currentDistribution.val / total) * 100),
      test: Math.round((currentDistribution.test / total) * 100),
    };
  }, [currentDistribution]);

  const handleImageSelect = (imageId: string) => {
    setSelectedImages(prev => 
      prev.includes(imageId) 
        ? prev.filter(id => id !== imageId)
        : [...prev, imageId]
    );
  };

  const handleSelectAll = (split?: 'train' | 'val' | 'test') => {
    if (split) {
      const splitImages = images.filter(img => img.dataset_split === split);
      setSelectedImages(splitImages.map(img => img.id));
    } else {
      setSelectedImages(images.map(img => img.id));
    }
  };

  const handleDeselectAll = () => {
    setSelectedImages([]);
  };

  const handleBulkSplitChange = () => {
    if (selectedImages.length > 0 && onSplitChange) {
      selectedImages.forEach(imageId => {
        onSplitChange(imageId, bulkSplit);
      });
      setSelectedImages([]);
    }
  };

  const handleIndividualSplitChange = (imageId: string, newSplit: 'train' | 'val' | 'test') => {
    onSplitChange?.(imageId, newSplit);
  };

  const splitColors = {
    train: { bg: 'bg-blue-100 dark:bg-blue-900', text: 'text-blue-800 dark:text-blue-200', border: 'border-blue-200 dark:border-blue-700' },
    val: { bg: 'bg-yellow-100 dark:bg-yellow-900', text: 'text-yellow-800 dark:text-yellow-200', border: 'border-yellow-200 dark:border-yellow-700' },
    test: { bg: 'bg-green-100 dark:bg-green-900', text: 'text-green-800 dark:text-green-200', border: 'border-green-200 dark:border-green-700' },
  };

  return (
    <div className={`dataset-split-manager space-y-6 ${className}`}>
      {/* Distribution Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Dataset Distribution
        </h3>
        
        {/* Visual Distribution Bar */}
        <div className="mb-4">
          <div className="flex h-8 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-600">
            <div 
              className="bg-blue-500 flex items-center justify-center text-white text-sm font-medium"
              style={{ width: `${percentages.train}%` }}
            >
              {percentages.train > 10 && `${percentages.train}%`}
            </div>
            <div 
              className="bg-yellow-500 flex items-center justify-center text-white text-sm font-medium"
              style={{ width: `${percentages.val}%` }}
            >
              {percentages.val > 10 && `${percentages.val}%`}
            </div>
            <div 
              className="bg-green-500 flex items-center justify-center text-white text-sm font-medium"
              style={{ width: `${percentages.test}%` }}
            >
              {percentages.test > 10 && `${percentages.test}%`}
            </div>
          </div>
        </div>

        {/* Distribution Stats */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          {(['train', 'val', 'test'] as const).map((split) => (
            <div key={split} className={`p-3 rounded-lg ${splitColors[split].bg}`}>
              <div className="text-center">
                <div className={`text-2xl font-bold ${splitColors[split].text}`}>
                  {currentDistribution[split]}
                </div>
                <div className={`text-sm font-medium ${splitColors[split].text}`}>
                  {split.toUpperCase()} ({percentages[split]}%)
                </div>
                <button
                  onClick={() => handleSelectAll(split)}
                  className={`text-xs mt-1 ${splitColors[split].text} hover:underline`}
                >
                  Select all
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Recommended Split */}
        <div className="text-sm text-gray-600 dark:text-gray-400">
          <p className="mb-1">
            <strong>Recommended split:</strong> 70% train, 20% validation, 10% test
          </p>
          <p>
            <strong>Current total:</strong> {currentDistribution.total} images
          </p>
        </div>
      </div>

      {/* Bulk Actions */}
      {selectedImages.length > 0 && (
        <div className="bg-blue-50 dark:bg-blue-950 rounded-lg p-4 border border-blue-200 dark:border-blue-700">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-blue-800 dark:text-blue-200 font-medium">
                {selectedImages.length} image{selectedImages.length !== 1 ? 's' : ''} selected
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <select
                value={bulkSplit}
                onChange={(e) => setBulkSplit(e.target.value as 'train' | 'val' | 'test')}
                className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
              >
                <option value="train">Training</option>
                <option value="val">Validation</option>
                <option value="test">Test</option>
              </select>
              <button
                onClick={handleBulkSplitChange}
                className="px-4 py-2 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
              >
                Move to {bulkSplit.toUpperCase()}
              </button>
              <button
                onClick={handleDeselectAll}
                className="px-3 py-2 text-sm bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 rounded transition-colors"
              >
                Deselect All
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
          Quick Actions
        </h4>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => handleSelectAll()}
            className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 rounded transition-colors"
          >
            Select All Images
          </button>
          <button
            onClick={() => {
              // Auto-split based on recommended ratios
              const shuffled = [...images].sort(() => Math.random() - 0.5);
              const trainCount = Math.floor(shuffled.length * 0.7);
              const valCount = Math.floor(shuffled.length * 0.2);
              
              shuffled.forEach((image, index) => {
                let newSplit: 'train' | 'val' | 'test';
                if (index < trainCount) {
                  newSplit = 'train';
                } else if (index < trainCount + valCount) {
                  newSplit = 'val';
                } else {
                  newSplit = 'test';
                }
                
                if (image.dataset_split !== newSplit) {
                  onSplitChange?.(image.id, newSplit);
                }
              });
            }}
            className="px-3 py-2 text-sm bg-blue-100 hover:bg-blue-200 dark:bg-blue-900 dark:hover:bg-blue-800 text-blue-800 dark:text-blue-200 rounded transition-colors"
          >
            Auto-Split (70/20/10)
          </button>
          <button
            onClick={() => {
              // Reset all to train
              images.forEach(image => {
                if (image.dataset_split !== 'train') {
                  onSplitChange?.(image.id, 'train');
                }
              });
            }}
            className="px-3 py-2 text-sm bg-yellow-100 hover:bg-yellow-200 dark:bg-yellow-900 dark:hover:bg-yellow-800 text-yellow-800 dark:text-yellow-200 rounded transition-colors"
          >
            Reset All to Train
          </button>
        </div>
      </div>

      {/* Image List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <h4 className="font-medium text-gray-900 dark:text-gray-100">
            Images ({images.length})
          </h4>
        </div>
        
        <div className="max-h-96 overflow-y-auto">
          {images.map((image) => (
            <div
              key={image.id}
              className={`flex items-center p-3 border-b border-gray-100 dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-750 ${
                selectedImages.includes(image.id) ? 'bg-blue-50 dark:bg-blue-950' : ''
              }`}
            >
              {/* Selection Checkbox */}
              <input
                type="checkbox"
                checked={selectedImages.includes(image.id)}
                onChange={() => handleImageSelect(image.id)}
                className="mr-3 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />

              {/* Thumbnail */}
              <div className="w-12 h-12 mr-3 bg-gray-100 dark:bg-gray-700 rounded overflow-hidden flex-shrink-0">
                <img
                  src={image.preview_url || image.file_path}
                  alt={image.original_filename}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIGZpbGw9IiNjY2MiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMTkgNWgtLjdMMTYgMmgtOGwtMi4zIDNINWMtMS4xIDAtMiAuOS0yIDJ2MTJjMCAxLjEuOSAyIDIgMmgxNGMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0wIDJWN0g1djE0aDE0VjdoMHptLTcgMWMxLjY1IDAgMyAxLjM1IDMgM3MtMS4zNSAzLTMgMy0zLTEuMzUtMy0zIDEuMzUtMyAzLTN6Ii8+PC9zdmc+';
                  }}
                />
              </div>

              {/* File Info */}
              <div className="flex-1 min-w-0 mr-3">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                  {image.original_filename}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {image.width && image.height ? `${image.width}×${image.height}` : 'Unknown size'}
                </p>
              </div>

              {/* Current Split */}
              <div className="mr-3">
                <span className={`px-2 py-1 rounded text-xs font-medium ${splitColors[image.dataset_split].bg} ${splitColors[image.dataset_split].text}`}>
                  {image.dataset_split.toUpperCase()}
                </span>
              </div>

              {/* Split Controls */}
              <div className="flex space-x-1">
                {(['train', 'val', 'test'] as const).map((split) => (
                  <button
                    key={split}
                    onClick={() => handleIndividualSplitChange(image.id, split)}
                    disabled={image.dataset_split === split}
                    className={`px-2 py-1 text-xs rounded transition-colors ${
                      image.dataset_split === split
                        ? `${splitColors[split].bg} ${splitColors[split].text} cursor-not-allowed`
                        : 'bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {split.charAt(0).toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};