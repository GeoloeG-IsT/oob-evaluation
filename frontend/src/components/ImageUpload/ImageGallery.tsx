'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { ImageGalleryProps, UploadedImage } from './types';
import { ImageCard } from './ImageCard';
import { ImagePreviewModal } from './ImagePreviewModal';
import { ImageFilters } from './ImageFilters';
import { calculateSplitDistribution } from './utils';

export const ImageGallery: React.FC<ImageGalleryProps> = ({
  images,
  onImageSelect,
  onImageDelete,
  selectedImages = [],
  allowMultipleSelect = false,
  showMetadata = true,
  filterByDatasetSplit = 'all',
  sortBy = 'upload_date',
  sortOrder = 'desc',
  className = '',
}) => {
  const [currentFilter, setCurrentFilter] = useState({
    datasetSplit: filterByDatasetSplit,
    sortBy,
    sortOrder,
  });
  const [selectedImageIds, setSelectedImageIds] = useState<string[]>(selectedImages);
  const [previewImage, setPreviewImage] = useState<UploadedImage | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [imagesPerPage] = useState(20);

  // Update selected images when prop changes
  useEffect(() => {
    setSelectedImageIds(selectedImages);
  }, [selectedImages]);

  // Calculate total counts for filters
  const totalCounts = useMemo(() => {
    const distribution = calculateSplitDistribution(images);
    return {
      all: distribution.total,
      train: distribution.train,
      val: distribution.val,
      test: distribution.test,
    };
  }, [images]);

  // Filter and sort images
  const processedImages = useMemo(() => {
    let filtered = [...images];

    // Filter by dataset split
    if (currentFilter.datasetSplit !== 'all') {
      filtered = filtered.filter(img => img.dataset_split === currentFilter.datasetSplit);
    }

    // Sort images
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (currentFilter.sortBy) {
        case 'filename':
          comparison = a.original_filename.localeCompare(b.original_filename);
          break;
        case 'file_size':
          comparison = a.file_size - b.file_size;
          break;
        case 'upload_date':
        default:
          comparison = new Date(a.upload_date).getTime() - new Date(b.upload_date).getTime();
          break;
      }

      return currentFilter.sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [images, currentFilter]);

  // Paginate images
  const paginatedImages = useMemo(() => {
    const startIndex = (currentPage - 1) * imagesPerPage;
    return processedImages.slice(startIndex, startIndex + imagesPerPage);
  }, [processedImages, currentPage, imagesPerPage]);

  const totalPages = Math.ceil(processedImages.length / imagesPerPage);

  const handleImageSelect = (image: UploadedImage) => {
    if (allowMultipleSelect) {
      const newSelected = selectedImageIds.includes(image.id)
        ? selectedImageIds.filter(id => id !== image.id)
        : [...selectedImageIds, image.id];
      setSelectedImageIds(newSelected);
    } else {
      setSelectedImageIds(selectedImageIds.includes(image.id) ? [] : [image.id]);
    }
    onImageSelect?.(image);
  };

  const handleImageDelete = async (imageId: string) => {
    // Remove from selected images if it was selected
    setSelectedImageIds(prev => prev.filter(id => id !== imageId));
    onImageDelete?.(imageId);
  };

  const handleFilterChange = (newFilter: Partial<typeof currentFilter>) => {
    setCurrentFilter(prev => ({ ...prev, ...newFilter }));
    setCurrentPage(1); // Reset to first page when filtering
  };

  const handlePreviewImage = (image: UploadedImage) => {
    setPreviewImage(image);
  };

  const handlePreviewNavigate = (direction: 'prev' | 'next') => {
    if (!previewImage) return;
    
    const currentIndex = processedImages.findIndex(img => img.id === previewImage.id);
    let newIndex;
    
    if (direction === 'prev') {
      newIndex = currentIndex > 0 ? currentIndex - 1 : processedImages.length - 1;
    } else {
      newIndex = currentIndex < processedImages.length - 1 ? currentIndex + 1 : 0;
    }
    
    setPreviewImage(processedImages[newIndex]);
  };

  if (images.length === 0) {
    return (
      <div className={`text-center py-12 ${className}`}>
        <div className="w-16 h-16 mx-auto text-gray-400 mb-4">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          No Images Found
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Upload some images to get started with your ML evaluation platform.
        </p>
      </div>
    );
  }

  return (
    <div className={`image-gallery ${className}`}>
      {/* Header with Filters */}
      <div className="mb-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Image Gallery
          </h2>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Showing {paginatedImages.length} of {processedImages.length} images
            {selectedImageIds.length > 0 && (
              <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full">
                {selectedImageIds.length} selected
              </span>
            )}
          </div>
        </div>

        <ImageFilters
          currentFilter={currentFilter}
          onFilterChange={handleFilterChange}
          totalCounts={totalCounts}
        />

        {/* Bulk Actions */}
        {selectedImageIds.length > 0 && (
          <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
            <span className="text-blue-800 dark:text-blue-200">
              {selectedImageIds.length} image{selectedImageIds.length !== 1 ? 's' : ''} selected
            </span>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedImageIds([])}
                className="px-3 py-1 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Deselect All
              </button>
              {onImageDelete && (
                <button
                  onClick={() => {
                    if (confirm(`Delete ${selectedImageIds.length} selected images?`)) {
                      selectedImageIds.forEach(id => onImageDelete(id));
                      setSelectedImageIds([]);
                    }
                  }}
                  className="px-3 py-1 text-sm bg-red-500 hover:bg-red-600 text-white rounded"
                >
                  Delete Selected
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Image Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-4 mb-6">
        {paginatedImages.map((image) => (
          <div key={image.id} className="relative">
            <ImageCard
              image={image}
              isSelected={selectedImageIds.includes(image.id)}
              onSelect={handleImageSelect}
              onDelete={onImageDelete ? handleImageDelete : undefined}
              showMetadata={showMetadata}
              className="h-full"
            />
            {/* Preview Button Overlay */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                handlePreviewImage(image);
              }}
              className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center opacity-0 hover:opacity-100"
              title="Preview image"
            >
              <div className="w-10 h-10 bg-white bg-opacity-90 rounded-full flex items-center justify-center">
                <svg className="w-5 h-5 text-gray-800" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                  />
                </svg>
              </div>
            </button>
          </div>
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center space-x-2">
          <button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            Previous
          </button>
          
          <div className="flex space-x-1">
            {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
              let pageNum;
              if (totalPages <= 7) {
                pageNum = i + 1;
              } else if (currentPage <= 4) {
                pageNum = i + 1;
              } else if (currentPage >= totalPages - 3) {
                pageNum = totalPages - 6 + i;
              } else {
                pageNum = currentPage - 3 + i;
              }
              
              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  className={`px-3 py-2 text-sm border rounded ${
                    currentPage === pageNum
                      ? 'bg-blue-500 text-white border-blue-500'
                      : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
          </div>
          
          <button
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            Next
          </button>
        </div>
      )}

      {/* Image Preview Modal */}
      <ImagePreviewModal
        image={previewImage}
        isOpen={!!previewImage}
        onClose={() => setPreviewImage(null)}
        onNavigate={handlePreviewNavigate}
        showMetadata={showMetadata}
      />
    </div>
  );
};