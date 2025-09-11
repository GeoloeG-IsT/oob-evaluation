'use client';

import React from 'react';
import { JobFiltersProps, JobStatus } from './types';

export const JobFilters: React.FC<JobFiltersProps> = ({
  statusFilter,
  typeFilter,
  dateRange,
  onStatusFilterChange,
  onTypeFilterChange,
  onDateRangeChange,
  onClearFilters,
  className = '',
}) => {
  const hasActiveFilters = statusFilter !== 'all' || typeFilter !== 'all' || dateRange !== null;

  const statusOptions: { value: JobStatus | 'all'; label: string; count?: number }[] = [
    { value: 'all', label: 'All Status' },
    { value: 'running', label: 'Running' },
    { value: 'pending', label: 'Pending' },
    { value: 'paused', label: 'Paused' },
    { value: 'completed', label: 'Completed' },
    { value: 'failed', label: 'Failed' },
    { value: 'cancelled', label: 'Cancelled' },
  ];

  const typeOptions: { value: 'training' | 'inference' | 'all'; label: string }[] = [
    { value: 'all', label: 'All Types' },
    { value: 'training', label: 'Training' },
    { value: 'inference', label: 'Inference' },
  ];

  const quickDateRanges = [
    { label: 'Last Hour', hours: 1 },
    { label: 'Last 24 Hours', hours: 24 },
    { label: 'Last 7 Days', hours: 24 * 7 },
    { label: 'Last 30 Days', hours: 24 * 30 },
  ];

  const handleQuickDateRange = (hours: number) => {
    const end = new Date().toISOString();
    const start = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();
    onDateRangeChange({ start, end });
  };

  return (
    <div className={`job-filters bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 ${className}`}>
      <div className="flex flex-wrap gap-4 items-center">
        {/* Status Filter */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Status:
          </label>
          <select
            value={statusFilter}
            onChange={(e) => onStatusFilterChange(e.target.value as JobStatus | 'all')}
            className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          >
            {statusOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Type Filter */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Type:
          </label>
          <select
            value={typeFilter}
            onChange={(e) => onTypeFilterChange(e.target.value as 'training' | 'inference' | 'all')}
            className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          >
            {typeOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Date Range Filter */}
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Date Range:
          </label>
          <div className="flex items-center space-x-2">
            {/* Quick Date Range Buttons */}
            <div className="flex space-x-1">
              {quickDateRanges.map((range) => (
                <button
                  key={range.label}
                  onClick={() => handleQuickDateRange(range.hours)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    dateRange &&
                    new Date(dateRange.end).getTime() - new Date(dateRange.start).getTime() === range.hours * 60 * 60 * 1000
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {range.label}
                </button>
              ))}
            </div>

            {/* Custom Date Range Inputs */}
            <div className="flex items-center space-x-1">
              <input
                type="datetime-local"
                value={dateRange?.start ? new Date(dateRange.start).toISOString().slice(0, 16) : ''}
                onChange={(e) => {
                  if (e.target.value) {
                    const start = new Date(e.target.value).toISOString();
                    const end = dateRange?.end || new Date().toISOString();
                    onDateRangeChange({ start, end });
                  }
                }}
                className="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="Start date"
              />
              <span className="text-gray-400">to</span>
              <input
                type="datetime-local"
                value={dateRange?.end ? new Date(dateRange.end).toISOString().slice(0, 16) : ''}
                onChange={(e) => {
                  if (e.target.value) {
                    const end = new Date(e.target.value).toISOString();
                    const start = dateRange?.start || new Date(0).toISOString();
                    onDateRangeChange({ start, end });
                  }
                }}
                className="px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="End date"
              />
            </div>
          </div>
        </div>

        {/* Clear Filters Button */}
        {hasActiveFilters && (
          <button
            onClick={onClearFilters}
            className="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 underline"
          >
            Clear All Filters
          </button>
        )}
      </div>

      {/* Active Filters Summary */}
      {hasActiveFilters && (
        <div className="mt-3 flex flex-wrap gap-2">
          {statusFilter !== 'all' && (
            <span className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full">
              Status: {statusFilter}
              <button
                onClick={() => onStatusFilterChange('all')}
                className="ml-1 text-blue-600 dark:text-blue-300 hover:text-blue-800 dark:hover:text-blue-100"
              >
                ×
              </button>
            </span>
          )}
          {typeFilter !== 'all' && (
            <span className="px-2 py-1 text-xs bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded-full">
              Type: {typeFilter}
              <button
                onClick={() => onTypeFilterChange('all')}
                className="ml-1 text-green-600 dark:text-green-300 hover:text-green-800 dark:hover:text-green-100"
              >
                ×
              </button>
            </span>
          )}
          {dateRange && (
            <span className="px-2 py-1 text-xs bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 rounded-full">
              Date: {new Date(dateRange.start).toLocaleDateString()} - {new Date(dateRange.end).toLocaleDateString()}
              <button
                onClick={() => onDateRangeChange(null)}
                className="ml-1 text-yellow-600 dark:text-yellow-300 hover:text-yellow-800 dark:hover:text-yellow-100"
              >
                ×
              </button>
            </span>
          )}
        </div>
      )}
    </div>
  );
};