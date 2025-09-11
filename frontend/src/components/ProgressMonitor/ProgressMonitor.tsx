'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { ProgressMonitorProps, TrainingJob, InferenceJob, JobStatus } from './types';
import { JobCard } from './JobCard';
import { JobFilters } from './JobFilters';
import { formatDuration, getJobProgress, sortJobsByStatus } from './utils';

export const ProgressMonitor: React.FC<ProgressMonitorProps> = ({
  jobs,
  onJobCancel,
  onJobPause,
  onJobResume,
  onJobDelete,
  autoRefreshInterval = 5000,
  maxVisibleJobs = 50,
  className = '',
}) => {
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set());
  const [filters, setFilters] = useState({
    statusFilter: 'all' as JobStatus | 'all',
    typeFilter: 'all' as 'training' | 'inference' | 'all',
    dateRange: null as { start: string; end: string } | null,
  });
  const [sortBy, setSortBy] = useState<'created_at' | 'status' | 'progress' | 'name'>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefreshInterval > 0) {
      const interval = setInterval(() => {
        // Trigger refresh via callback or websocket
        console.log('Auto-refreshing jobs...');
      }, autoRefreshInterval);

      return () => clearInterval(interval);
    }
  }, [autoRefreshInterval]);

  // Filter and sort jobs
  const processedJobs = useMemo(() => {
    let filtered = [...jobs];

    // Apply status filter
    if (filters.statusFilter !== 'all') {
      filtered = filtered.filter(job => job.status === filters.statusFilter);
    }

    // Apply type filter
    if (filters.typeFilter !== 'all') {
      const isTraining = filters.typeFilter === 'training';
      filtered = filtered.filter(job => 
        isTraining ? 'total_epochs' in job : 'total_images' in job
      );
    }

    // Apply date range filter
    if (filters.dateRange) {
      const startDate = new Date(filters.dateRange.start);
      const endDate = new Date(filters.dateRange.end);
      filtered = filtered.filter(job => {
        const jobDate = new Date(job.created_at);
        return jobDate >= startDate && jobDate <= endDate;
      });
    }

    // Sort jobs
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'status':
          comparison = a.status.localeCompare(b.status);
          break;
        case 'progress':
          comparison = getJobProgress(a) - getJobProgress(b);
          break;
        case 'created_at':
        default:
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    // Limit visible jobs
    return filtered.slice(0, maxVisibleJobs);
  }, [jobs, filters, sortBy, sortOrder, maxVisibleJobs]);

  // Group jobs by status for summary
  const jobSummary = useMemo(() => {
    return jobs.reduce((acc, job) => {
      acc[job.status] = (acc[job.status] || 0) + 1;
      return acc;
    }, {} as Record<JobStatus, number>);
  }, [jobs]);

  const handleJobExpand = (jobId: string) => {
    setExpandedJobs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(jobId)) {
        newSet.delete(jobId);
      } else {
        newSet.add(jobId);
      }
      return newSet;
    });
  };

  const getStatusColor = (status: JobStatus) => {
    switch (status) {
      case 'running':
        return 'text-blue-600 dark:text-blue-400';
      case 'completed':
        return 'text-green-600 dark:text-green-400';
      case 'failed':
        return 'text-red-600 dark:text-red-400';
      case 'cancelled':
        return 'text-gray-600 dark:text-gray-400';
      case 'paused':
        return 'text-yellow-600 dark:text-yellow-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  return (
    <div className={`progress-monitor ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Job Progress Monitor
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Real-time monitoring of training and inference jobs
          </p>
        </div>

        {/* Summary Stats */}
        <div className="flex items-center space-x-4 text-sm">
          {Object.entries(jobSummary).map(([status, count]) => (
            <div key={status} className="flex items-center space-x-1">
              <div className={`w-3 h-3 rounded-full ${
                status === 'running' ? 'bg-blue-500' :
                status === 'completed' ? 'bg-green-500' :
                status === 'failed' ? 'bg-red-500' :
                status === 'paused' ? 'bg-yellow-500' :
                'bg-gray-500'
              }`} />
              <span className={`font-medium ${getStatusColor(status as JobStatus)}`}>
                {count}
              </span>
              <span className="text-gray-500 dark:text-gray-400 capitalize">
                {status}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Filters */}
      <div className="mb-6">
        <JobFilters
          statusFilter={filters.statusFilter}
          typeFilter={filters.typeFilter}
          dateRange={filters.dateRange}
          onStatusFilterChange={(status) => setFilters(prev => ({ ...prev, statusFilter: status }))}
          onTypeFilterChange={(type) => setFilters(prev => ({ ...prev, typeFilter: type }))}
          onDateRangeChange={(range) => setFilters(prev => ({ ...prev, dateRange: range }))}
          onClearFilters={() => setFilters({
            statusFilter: 'all',
            typeFilter: 'all',
            dateRange: null,
          })}
        />
      </div>

      {/* Sort Controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Sort by:
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              <option value="created_at">Created Date</option>
              <option value="status">Status</option>
              <option value="progress">Progress</option>
              <option value="name">Name</option>
            </select>
          </div>

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

        <div className="text-sm text-gray-600 dark:text-gray-400">
          Showing {processedJobs.length} of {jobs.length} jobs
          {maxVisibleJobs < jobs.length && jobs.length > maxVisibleJobs && (
            <span className="ml-1">(limited to {maxVisibleJobs})</span>
          )}
        </div>
      </div>

      {/* Jobs List */}
      {processedJobs.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 mx-auto mb-4 text-gray-400">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No Jobs Found
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            {filters.statusFilter !== 'all' || filters.typeFilter !== 'all' || filters.dateRange
              ? 'No jobs match your current filters.'
              : 'No jobs are currently running or scheduled.'
            }
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {processedJobs.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              isExpanded={expandedJobs.has(job.id)}
              onExpand={handleJobExpand}
              onCancel={onJobCancel}
              onPause={onJobPause}
              onResume={onJobResume}
              onDelete={onJobDelete}
            />
          ))}
        </div>
      )}

      {/* Auto-refresh indicator */}
      {autoRefreshInterval > 0 && (
        <div className="mt-4 text-center">
          <div className="inline-flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span>Auto-refreshing every {formatDuration(autoRefreshInterval)}</span>
          </div>
        </div>
      )}
    </div>
  );
};