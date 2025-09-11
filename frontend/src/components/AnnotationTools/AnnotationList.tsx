'use client';

import React, { useState, useMemo } from 'react';
import { AnnotationListProps } from './types';
import { formatFileSize } from '../ImageUpload/utils';

export const AnnotationList: React.FC<AnnotationListProps> = ({
  annotations,
  categories,
  onAnnotationSelect,
  onAnnotationDelete,
  onAnnotationUpdate,
  selectedAnnotationId,
  showConfidence = true,
  className = '',
}) => {
  const [sortBy, setSortBy] = useState<'category' | 'type' | 'confidence' | 'created_at'>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [filterType, setFilterType] = useState<string>('all');

  // Sort and filter annotations
  const processedAnnotations = useMemo(() => {
    let filtered = [...annotations];

    // Filter by category
    if (filterCategory !== 'all') {
      filtered = filtered.filter(ann => ann.category_id === filterCategory);
    }

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter(ann => ann.type === filterType);
    }

    // Sort annotations
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'category':
          comparison = a.category_name.localeCompare(b.category_name);
          break;
        case 'type':
          comparison = a.type.localeCompare(b.type);
          break;
        case 'confidence':
          comparison = (a.confidence || 0) - (b.confidence || 0);
          break;
        case 'created_at':
        default:
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [annotations, sortBy, sortOrder, filterCategory, filterType]);

  const handleDeleteAnnotation = (annotationId: string) => {
    if (confirm('Are you sure you want to delete this annotation?')) {
      onAnnotationDelete?.(annotationId);
    }
  };

  const formatAreaValue = (area?: number): string => {
    if (!area) return 'N/A';
    return area < 1000 ? `${Math.round(area)}px²` : `${(area / 1000).toFixed(1)}k px²`;
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'bbox':
        return '▢';
      case 'polygon':
        return '⬟';
      case 'point':
        return '•';
      case 'circle':
        return '○';
      default:
        return '?';
    }
  };

  const getCreatedByIcon = (createdBy: string) => {
    return createdBy === 'user' ? '👤' : '🤖';
  };

  return (
    <div className={`annotation-list bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Annotations ({processedAnnotations.length})
          </h3>
          
          {/* Sort Controls */}
          <div className="flex items-center space-x-2 text-sm">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="created_at">Date</option>
              <option value="category">Category</option>
              <option value="type">Type</option>
              {showConfidence && <option value="confidence">Confidence</option>}
            </select>
            
            <button
              onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
              className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d={sortOrder === 'asc' 
                    ? "M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z"
                    : "M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                  }
                  clipRule="evenodd"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-3 text-sm">
          <div className="flex items-center space-x-2">
            <label className="text-gray-600 dark:text-gray-400">Category:</label>
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="all">All</option>
              {categories.map(category => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <label className="text-gray-600 dark:text-gray-400">Type:</label>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="all">All</option>
              <option value="bbox">Bounding Box</option>
              <option value="polygon">Polygon</option>
              <option value="point">Point</option>
              <option value="circle">Circle</option>
            </select>
          </div>
        </div>
      </div>

      {/* Annotations List */}
      <div className="max-h-96 overflow-y-auto">
        {processedAnnotations.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            <div className="w-12 h-12 mx-auto mb-3 text-gray-400">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <p className="text-lg font-medium mb-1">No Annotations</p>
            <p className="text-sm">Start annotating the image to see annotations here.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {processedAnnotations.map((annotation, index) => {
              const category = categories.find(cat => cat.id === annotation.category_id);
              const isSelected = annotation.id === selectedAnnotationId;
              
              return (
                <div
                  key={annotation.id}
                  className={`p-4 cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-blue-50 dark:bg-blue-950 border-l-4 border-blue-500'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-750'
                  }`}
                  onClick={() => onAnnotationSelect?.(annotation)}
                >
                  <div className="flex items-start justify-between">
                    {/* Annotation Info */}
                    <div className="flex items-start space-x-3 flex-1">
                      {/* Color & Type Indicator */}
                      <div className="flex flex-col items-center space-y-1">
                        <div
                          className="w-4 h-4 rounded border border-white shadow-sm"
                          style={{ backgroundColor: category?.color || '#gray' }}
                        />
                        <span className="text-sm" title={annotation.type}>
                          {getTypeIcon(annotation.type)}
                        </span>
                      </div>

                      {/* Details */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <h4 className="font-medium text-gray-900 dark:text-gray-100">
                            {annotation.category_name}
                          </h4>
                          <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded">
                            #{index + 1}
                          </span>
                          <span className="text-xs" title={`Created by ${annotation.created_by}`}>
                            {getCreatedByIcon(annotation.created_by)}
                          </span>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600 dark:text-gray-400">
                          <div>
                            <span className="font-medium">Type:</span> {annotation.type}
                          </div>
                          <div>
                            <span className="font-medium">Area:</span> {formatAreaValue(annotation.area)}
                          </div>
                          {showConfidence && annotation.confidence && (
                            <div>
                              <span className="font-medium">Confidence:</span>{' '}
                              <span className={`font-medium ${
                                annotation.confidence > 0.8 ? 'text-green-600 dark:text-green-400' :
                                annotation.confidence > 0.5 ? 'text-yellow-600 dark:text-yellow-400' :
                                'text-red-600 dark:text-red-400'
                              }`}>
                                {(annotation.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          )}
                          {annotation.model_name && (
                            <div>
                              <span className="font-medium">Model:</span> {annotation.model_name}
                            </div>
                          )}
                        </div>

                        {/* Coordinates Summary */}
                        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                          {annotation.type === 'bbox' && annotation.bbox && (
                            <span>
                              Box: ({Math.round(annotation.bbox.x)}, {Math.round(annotation.bbox.y)}) 
                              {Math.round(annotation.bbox.width)}×{Math.round(annotation.bbox.height)}
                            </span>
                          )}
                          {annotation.type === 'point' && annotation.center && (
                            <span>
                              Point: ({Math.round(annotation.center.x)}, {Math.round(annotation.center.y)})
                            </span>
                          )}
                          {annotation.type === 'circle' && annotation.center && annotation.radius && (
                            <span>
                              Circle: center ({Math.round(annotation.center.x)}, {Math.round(annotation.center.y)}), 
                              radius {Math.round(annotation.radius)}
                            </span>
                          )}
                          {annotation.type === 'polygon' && annotation.segmentation && (
                            <span>
                              Polygon: {annotation.segmentation.points.length} points
                            </span>
                          )}
                        </div>

                        {/* Timestamps */}
                        <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Created: {new Date(annotation.created_at).toLocaleString()}
                          {annotation.updated_at !== annotation.created_at && (
                            <span className="ml-2">
                              • Updated: {new Date(annotation.updated_at).toLocaleString()}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      {onAnnotationUpdate && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            // This would open an edit dialog or inline editor
                            console.log('Edit annotation:', annotation.id);
                          }}
                          className="p-1 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400"
                          title="Edit annotation"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                        </button>
                      )}
                      
                      {onAnnotationDelete && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteAnnotation(annotation.id);
                          }}
                          className="p-1 text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                          title="Delete annotation"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      )}

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          navigator.clipboard.writeText(JSON.stringify(annotation, null, 2));
                          // You might want to show a toast notification here
                        }}
                        className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                        title="Copy annotation data"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Summary Footer */}
      {processedAnnotations.length > 0 && (
        <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-750">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            <div className="flex justify-between items-center">
              <div className="flex space-x-4">
                <span>User: {processedAnnotations.filter(a => a.created_by === 'user').length}</span>
                <span>AI: {processedAnnotations.filter(a => a.created_by === 'model').length}</span>
              </div>
              {showConfidence && (
                <div>
                  Avg Confidence: {
                    processedAnnotations.filter(a => a.confidence).length > 0
                      ? (processedAnnotations.reduce((sum, a) => sum + (a.confidence || 0), 0) / 
                         processedAnnotations.filter(a => a.confidence).length * 100).toFixed(1)
                      : 'N/A'
                  }%
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};