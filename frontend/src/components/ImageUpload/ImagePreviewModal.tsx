'use client';

import React, { useEffect, useCallback } from 'react';
import { ImagePreviewModalProps } from './types';
import { formatFileSize, formatUploadDate } from './utils';

export const ImagePreviewModal: React.FC<ImagePreviewModalProps> = ({
  image,
  isOpen,
  onClose,
  onNavigate,
  showMetadata = true,
}) => {
  const handleKeyPress = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen) return;
      
      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          onNavigate?.('prev');
          break;
        case 'ArrowRight':
          onNavigate?.('next');
          break;
      }
    },
    [isOpen, onClose, onNavigate]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  if (!isOpen || !image) return null;

  const datasetSplitColors = {
    train: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    val: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    test: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-75 transition-opacity"
        onClick={onClose}
      />
      
      {/* Modal Content */}
      <div className="relative max-w-7xl max-h-full w-full h-full flex flex-col lg:flex-row bg-white dark:bg-gray-900 shadow-2xl">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 w-10 h-10 bg-black bg-opacity-50 hover:bg-opacity-75 text-white rounded-full flex items-center justify-center transition-all"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Navigation Buttons */}
        {onNavigate && (
          <>
            <button
              onClick={() => onNavigate('prev')}
              className="absolute left-4 top-1/2 transform -translate-y-1/2 z-10 w-12 h-12 bg-black bg-opacity-50 hover:bg-opacity-75 text-white rounded-full flex items-center justify-center transition-all"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <button
              onClick={() => onNavigate('next')}
              className="absolute right-4 top-1/2 transform -translate-y-1/2 z-10 w-12 h-12 bg-black bg-opacity-50 hover:bg-opacity-75 text-white rounded-full flex items-center justify-center transition-all"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </>
        )}

        {/* Image Container */}
        <div className="flex-1 flex items-center justify-center p-8 bg-gray-100 dark:bg-gray-800 min-h-0">
          <img
            src={image.file_path}
            alt={image.original_filename}
            className="max-w-full max-h-full object-contain"
          />
        </div>

        {/* Metadata Panel */}
        {showMetadata && (
          <div className="lg:w-80 lg:flex-shrink-0 bg-white dark:bg-gray-900 border-t lg:border-t-0 lg:border-l border-gray-200 dark:border-gray-700 overflow-y-auto">
            <div className="p-6 space-y-6">
              {/* Header */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 break-all">
                  {image.original_filename}
                </h3>
                <div className="mt-2 flex items-center justify-between">
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      datasetSplitColors[image.dataset_split]
                    }`}
                  >
                    {image.dataset_split.toUpperCase()} SET
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {formatUploadDate(image.upload_date)}
                  </span>
                </div>
              </div>

              {/* Basic Info */}
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100">
                  Image Information
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Format:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100 uppercase">
                      {image.format}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">File Size:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {formatFileSize(image.file_size)}
                    </span>
                  </div>
                  {image.width && image.height && (
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Dimensions:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {image.width} × {image.height}
                      </span>
                    </div>
                  )}
                  {image.channels && (
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Channels:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {image.channels}
                      </span>
                    </div>
                  )}
                  {image.color_mode && (
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Color Mode:</span>
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {image.color_mode}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Technical Details */}
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900 dark:text-gray-100">
                  Technical Details
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Image ID:</span>
                    <span className="font-mono text-xs text-gray-900 dark:text-gray-100 break-all">
                      {image.id}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Filename:</span>
                    <span className="font-mono text-xs text-gray-900 dark:text-gray-100 break-all">
                      {image.filename}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Path:</span>
                    <span className="font-mono text-xs text-gray-900 dark:text-gray-100 break-all">
                      {image.file_path}
                    </span>
                  </div>
                </div>
              </div>

              {/* Additional Metadata */}
              {image.metadata && Object.keys(image.metadata).length > 0 && (
                <div className="space-y-4">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100">
                    Additional Metadata
                  </h4>
                  <div className="space-y-2 text-sm">
                    {Object.entries(image.metadata).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400 capitalize">
                          {key.replace(/[_-]/g, ' ')}:
                        </span>
                        <span className="font-medium text-gray-900 dark:text-gray-100 break-all text-right">
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="space-y-2 pt-4 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(image.file_path);
                    // You might want to show a toast notification here
                  }}
                  className="w-full px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-lg transition-colors"
                >
                  Copy File Path
                </button>
                <a
                  href={image.file_path}
                  download={image.original_filename}
                  className="w-full px-4 py-2 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors text-center block"
                >
                  Download Original
                </a>
              </div>

              {/* Keyboard Shortcuts */}
              <div className="text-xs text-gray-500 dark:text-gray-400 pt-4 border-t border-gray-200 dark:border-gray-700">
                <p className="mb-2">Keyboard shortcuts:</p>
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Close:</span>
                    <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">Esc</kbd>
                  </div>
                  {onNavigate && (
                    <>
                      <div className="flex justify-between">
                        <span>Previous:</span>
                        <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">←</kbd>
                      </div>
                      <div className="flex justify-between">
                        <span>Next:</span>
                        <kbd className="px-2 py-1 bg-gray-100 dark:bg-gray-800 rounded">→</kbd>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};