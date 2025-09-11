'use client';

import React, { useState } from 'react';
import { ImageCardProps } from './types';
import { formatFileSize, formatUploadDate } from './utils';

export const ImageCard: React.FC<ImageCardProps> = ({
  image,
  isSelected = false,
  onSelect,
  onDelete,
  showMetadata = true,
  className = '',
}) => {
  const [imageError, setImageError] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleImageError = () => {
    setImageError(true);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (showDeleteConfirm) {
      onDelete?.(image.id);
      setShowDeleteConfirm(false);
    } else {
      setShowDeleteConfirm(true);
    }
  };

  const handleSelect = () => {
    onSelect?.(image);
  };

  const datasetSplitColors = {
    train: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    val: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    test: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  };

  return (
    <div
      className={`
        relative group cursor-pointer transition-all duration-200 rounded-lg overflow-hidden
        ${isSelected
          ? 'ring-2 ring-blue-500 ring-offset-2 dark:ring-offset-gray-800'
          : 'hover:shadow-lg'
        }
        ${className}
      `}
      onClick={handleSelect}
    >
      {/* Selection Indicator */}
      {isSelected && (
        <div className="absolute top-2 left-2 z-10 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      )}

      {/* Delete Button */}
      {onDelete && (
        <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleDelete}
            className={`
              w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm
              transition-colors ${
                showDeleteConfirm
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-black bg-opacity-50 hover:bg-red-500'
              }
            `}
            title={showDeleteConfirm ? 'Click again to confirm' : 'Delete image'}
          >
            {showDeleteConfirm ? '✓' : '×'}
          </button>
        </div>
      )}

      {/* Image */}
      <div className="aspect-square bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
        {!imageError ? (
          <img
            src={image.preview_url || image.file_path}
            alt={image.original_filename}
            className="w-full h-full object-cover"
            onError={handleImageError}
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-gray-400 p-4">
            <svg className="w-8 h-8 mb-2" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-xs text-center">Image not available</span>
          </div>
        )}
      </div>

      {/* Metadata Overlay */}
      {showMetadata && (
        <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="absolute bottom-0 left-0 right-0 p-3 text-white">
            <div className="space-y-1">
              <p className="font-medium text-sm truncate" title={image.original_filename}>
                {image.original_filename}
              </p>
              <div className="flex items-center justify-between text-xs">
                <span>{formatFileSize(image.file_size)}</span>
                <span className="flex items-center">
                  {image.width && image.height && (
                    <span className="mr-2">{image.width} × {image.height}</span>
                  )}
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${
                      datasetSplitColors[image.dataset_split]
                    }`}
                  >
                    {image.dataset_split.toUpperCase()}
                  </span>
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Metadata Panel (Always Visible) */}
      {showMetadata && (
        <div className="p-3 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-sm text-gray-900 dark:text-gray-100 truncate mr-2">
                {image.original_filename}
              </h4>
              <span
                className={`px-2 py-1 rounded-full text-xs font-medium whitespace-nowrap ${
                  datasetSplitColors[image.dataset_split]
                }`}
              >
                {image.dataset_split.toUpperCase()}
              </span>
            </div>

            <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <span>{formatFileSize(image.file_size)}</span>
              {image.width && image.height && (
                <span>{image.width} × {image.height}</span>
              )}
            </div>

            <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <span className="uppercase">{image.format}</span>
              <span>{formatUploadDate(image.upload_date)}</span>
            </div>

            {image.metadata && Object.keys(image.metadata).length > 0 && (
              <details className="text-xs">
                <summary className="cursor-pointer text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200">
                  Additional metadata
                </summary>
                <div className="mt-1 p-2 bg-gray-50 dark:bg-gray-900 rounded text-gray-700 dark:text-gray-300">
                  {Object.entries(image.metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="font-medium">{key}:</span>
                      <span>{String(value)}</span>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        </div>
      )}

      {/* Delete Confirmation Overlay */}
      {showDeleteConfirm && (
        <div className="absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center z-20">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 max-w-xs text-center">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
              Delete Image?
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              This action cannot be undone.
            </p>
            <div className="flex space-x-2 justify-center">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowDeleteConfirm(false);
                }}
                className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 rounded"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="px-3 py-1 text-sm bg-red-500 hover:bg-red-600 text-white rounded"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};