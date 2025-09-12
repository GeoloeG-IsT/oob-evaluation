'use client';

import React from 'react';
import { ImageFiltersProps } from './types';

export const ImageFilters: React.FC<ImageFiltersProps> = ({
  currentFilter,
  onFilterChange,
  totalCounts,
  className = '',
}) => {
  return (
    <div className={`flex flex-wrap gap-4 items-center ${className}`}>
      {/* Dataset Split Filter */}
      <div className="flex items-center space-x-2">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Dataset:
        </label>
        <div className="flex space-x-1">
          {(['all', 'train', 'val', 'test'] as const).map((split) => (
            <button
              key={split}
              onClick={() => onFilterChange({ datasetSplit: split })}
              className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                currentFilter.datasetSplit === split
                  ? split === 'all'
                    ? 'bg-gray-600 text-white'
                    : split === 'train'
                    ? 'bg-blue-600 text-white'
                    : split === 'val'
                    ? 'bg-yellow-600 text-white'
                    : 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              {split === 'all' ? 'All' : split.toUpperCase()} ({totalCounts[split]})
            </button>
          ))}
        </div>
      </div>

      {/* Sort Options */}
      <div className="flex items-center space-x-2">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Sort by:
        </label>
        <select
          value={currentFilter.sortBy}
          onChange={(e) => onFilterChange({ sortBy: e.target.value as 'upload_date' | 'filename' | 'file_size' })}
          className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        >
          <option value="upload_date">Upload Date</option>
          <option value="filename">Filename</option>
          <option value="file_size">File Size</option>
        </select>
      </div>

      {/* Sort Order */}
      <div className="flex items-center space-x-2">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Order:
        </label>
        <div className="flex space-x-1">
          <button
            onClick={() => onFilterChange({ sortOrder: 'desc' })}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              currentFilter.sortOrder === 'desc'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
          <button
            onClick={() => onFilterChange({ sortOrder: 'asc' })}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              currentFilter.sortOrder === 'asc'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Reset Filters */}
      {(currentFilter.datasetSplit !== 'all' || 
        currentFilter.sortBy !== 'upload_date' || 
        currentFilter.sortOrder !== 'desc') && (
        <button
          onClick={() => onFilterChange({ 
            datasetSplit: 'all', 
            sortBy: 'upload_date', 
            sortOrder: 'desc' 
          })}
          className="px-3 py-1 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 underline"
        >
          Reset Filters
        </button>
      )}
    </div>
  );
};