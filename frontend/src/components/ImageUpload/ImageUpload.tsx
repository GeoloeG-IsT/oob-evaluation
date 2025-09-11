'use client';

import React, { useCallback, useState, useRef } from 'react';
import { ImageUploadProps, UploadProgress, UploadedImage } from './types';
import { validateFile, uploadImageToServer, createImagePreview } from './utils';

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onUploadComplete,
  onUploadProgress,
  acceptedFormats = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff'],
  maxFileSize = 100 * 1024 * 1024, // 100MB
  allowMultiple = true,
  defaultDatasetSplit = 'train',
  className = '',
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [previews, setPreviews] = useState<{ file: File; preview: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(async (files: FileList) => {
    const validFiles: File[] = [];
    const errors: string[] = [];

    // Validate files
    Array.from(files).forEach((file) => {
      const validation = validateFile(file);
      if (validation.valid) {
        validFiles.push(file);
      } else {
        errors.push(`${file.name}: ${validation.error}`);
      }
    });

    if (errors.length > 0) {
      console.error('File validation errors:', errors);
      // You might want to show these errors in the UI
    }

    if (validFiles.length === 0) return;

    // Create previews
    const newPreviews = await Promise.all(
      validFiles.map(async (file) => ({
        file,
        preview: await createImagePreview(file),
      }))
    );
    setPreviews(newPreviews);

    // Initialize progress tracking
    const initialProgress = validFiles.map((file) => ({
      filename: file.name,
      progress: 0,
      status: 'uploading' as const,
    }));
    setUploadProgress(initialProgress);
    onUploadProgress?.(initialProgress);

    setIsUploading(true);

    try {
      // Upload files
      const uploadPromises = validFiles.map(async (file, index) => {
        try {
          const result = await uploadImageToServer(
            file,
            defaultDatasetSplit,
            (progress) => {
              setUploadProgress((prev) => {
                const updated = [...prev];
                updated[index] = {
                  ...updated[index],
                  progress,
                  status: progress === 100 ? 'processing' : 'uploading',
                };
                onUploadProgress?.(updated);
                return updated;
              });
            }
          );

          // Update progress to completed
          setUploadProgress((prev) => {
            const updated = [...prev];
            updated[index] = {
              ...updated[index],
              progress: 100,
              status: 'completed',
            };
            onUploadProgress?.(updated);
            return updated;
          });

          return result.image;
        } catch (error) {
          // Update progress to error
          setUploadProgress((prev) => {
            const updated = [...prev];
            updated[index] = {
              ...updated[index],
              status: 'error',
              error: error instanceof Error ? error.message : 'Upload failed',
            };
            onUploadProgress?.(updated);
            return updated;
          });
          return null;
        }
      });

      const results = await Promise.all(uploadPromises);
      const successfulUploads = results.filter(Boolean) as UploadedImage[];

      if (successfulUploads.length > 0) {
        onUploadComplete?.(successfulUploads);
      }
    } finally {
      setIsUploading(false);
      // Clear previews after upload
      setTimeout(() => {
        setPreviews([]);
        setUploadProgress([]);
      }, 3000);
    }
  }, [defaultDatasetSplit, onUploadComplete, onUploadProgress]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        handleFiles(files);
      }
    },
    [handleFiles]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFiles(files);
      }
    },
    [handleFiles]
  );

  const openFileDialog = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className={`image-upload-container ${className}`}>
      {/* Main Upload Area */}
      <div
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200
          ${isDragging
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
            : 'border-gray-300 hover:border-gray-400'
          }
          ${isUploading ? 'opacity-50 pointer-events-none' : 'cursor-pointer'}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedFormats.join(',')}
          multiple={allowMultiple}
          onChange={handleFileInput}
          className="hidden"
        />

        <div className="space-y-4">
          {!isUploading ? (
            <>
              <div className="w-16 h-16 mx-auto text-gray-400">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
                  Drop images here or click to browse
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                  Supports: {acceptedFormats.map(f => f.split('/')[1].toUpperCase()).join(', ')}
                  <br />
                  Max size: {Math.round(maxFileSize / (1024 * 1024))}MB
                  {allowMultiple && ' • Multiple files allowed'}
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="w-16 h-16 mx-auto text-blue-500 animate-spin">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
              </div>
              <p className="text-lg font-medium text-blue-600 dark:text-blue-400">
                Uploading {uploadProgress.length} image{uploadProgress.length !== 1 ? 's' : ''}...
              </p>
            </>
          )}
        </div>
      </div>

      {/* Dataset Split Selector */}
      <div className="mt-4 flex items-center justify-center space-x-4 text-sm">
        <label className="text-gray-700 dark:text-gray-300">Upload to:</label>
        <select
          value={defaultDatasetSplit}
          onChange={(e) => {
            // This would need to be passed as a prop to update the parent component
            console.log('Dataset split changed to:', e.target.value);
          }}
          className="px-3 py-1 border border-gray-300 rounded-md bg-white dark:bg-gray-800 dark:border-gray-600"
          disabled={isUploading}
        >
          <option value="train">Training Set</option>
          <option value="val">Validation Set</option>
          <option value="test">Test Set</option>
        </select>
      </div>

      {/* Upload Progress */}
      {uploadProgress.length > 0 && (
        <div className="mt-6 space-y-3">
          <h4 className="font-medium text-gray-900 dark:text-gray-100">
            Upload Progress ({uploadProgress.filter(p => p.status === 'completed').length}/{uploadProgress.length})
          </h4>
          {uploadProgress.map((progress, index) => (
            <div key={index} className="space-y-2">
              <div className="flex justify-between items-center text-sm">
                <span className="truncate max-w-xs text-gray-700 dark:text-gray-300">
                  {progress.filename}
                </span>
                <span
                  className={`px-2 py-1 rounded text-xs font-medium ${
                    progress.status === 'completed'
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : progress.status === 'error'
                      ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                      : progress.status === 'processing'
                      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                  }`}
                >
                  {progress.status === 'completed'
                    ? 'Complete'
                    : progress.status === 'error'
                    ? 'Error'
                    : progress.status === 'processing'
                    ? 'Processing'
                    : `${Math.round(progress.progress)}%`}
                </span>
              </div>
              {progress.status !== 'error' && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      progress.status === 'completed'
                        ? 'bg-green-500'
                        : progress.status === 'processing'
                        ? 'bg-yellow-500'
                        : 'bg-blue-500'
                    }`}
                    style={{
                      width: `${progress.status === 'processing' || progress.status === 'completed' ? 100 : progress.progress}%`,
                    }}
                  />
                </div>
              )}
              {progress.error && (
                <p className="text-sm text-red-600 dark:text-red-400">{progress.error}</p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Image Previews */}
      {previews.length > 0 && (
        <div className="mt-6">
          <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
            Preview ({previews.length} image{previews.length !== 1 ? 's' : ''})
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {previews.map((preview, index) => (
              <div key={index} className="relative group">
                <img
                  src={preview.preview}
                  alt={`Preview ${index + 1}`}
                  className="w-full h-24 object-cover rounded-lg border border-gray-200 dark:border-gray-700"
                />
                <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                  <span className="text-white text-xs font-medium px-2 py-1 bg-black bg-opacity-75 rounded">
                    {preview.file.name.length > 15
                      ? `${preview.file.name.substring(0, 12)}...`
                      : preview.file.name}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};