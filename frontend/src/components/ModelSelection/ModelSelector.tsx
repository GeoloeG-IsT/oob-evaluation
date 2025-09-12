'use client';

import React, { useState, useMemo } from 'react';
import { ModelSelectorProps } from './types';
import { ModelCard } from './ModelCard';
import { getModelArchitectureInfo } from './utils';

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModelId,
  onModelSelect,
  onModelCreate,
  onModelDelete,
  filterByType = 'all',
  filterByArchitecture = 'all',
  showPerformance = true,
  isReadOnly = false,
  className = '',
}) => {
  const [currentTypeFilter, setCurrentTypeFilter] = useState(filterByType);
  const [currentArchFilter, setCurrentArchFilter] = useState(filterByArchitecture);
  const [sortBy, setSortBy] = useState<'name' | 'type' | 'architecture' | 'performance' | 'created'>('created');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [searchQuery, setSearchQuery] = useState('');

  // Get unique architectures and types from models
  const availableArchitectures = useMemo(() => {
    const architectures = new Set(models.map(m => m.architecture));
    return Array.from(architectures).sort();
  }, [models]);

  const availableTypes = useMemo(() => {
    const types = new Set(models.map(m => m.type));
    return Array.from(types).sort();
  }, [models]);

  // Filter and sort models
  const processedModels = useMemo(() => {
    let filtered = [...models];

    // Apply filters
    if (currentTypeFilter !== 'all') {
      filtered = filtered.filter(model => model.type === currentTypeFilter);
    }

    if (currentArchFilter !== 'all') {
      filtered = filtered.filter(model => model.architecture === currentArchFilter);
    }

    // Apply search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim();
      filtered = filtered.filter(model => 
        model.name.toLowerCase().includes(query) ||
        model.architecture.toLowerCase().includes(query) ||
        model.variant.toLowerCase().includes(query) ||
        model.description?.toLowerCase().includes(query)
      );
    }

    // Sort models
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'type':
          comparison = a.type.localeCompare(b.type);
          break;
        case 'architecture':
          comparison = a.architecture.localeCompare(b.architecture);
          break;
        case 'performance':
          const aMap = a.performance_metrics?.map || 0;
          const bMap = b.performance_metrics?.map || 0;
          comparison = aMap - bMap;
          break;
        case 'created':
        default:
          comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [models, currentTypeFilter, currentArchFilter, searchQuery, sortBy, sortOrder]);

  const handleCreateModel = () => {
    onModelCreate?.();
  };

  return (
    <div className={`model-selector ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Model Selection
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Choose a model for training or inference ({processedModels.length} available)
          </p>
        </div>

        {!isReadOnly && onModelCreate && (
          <button
            onClick={handleCreateModel}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
          >
            Add Model
          </button>
        )}
      </div>

      {/* Filters and Search */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 mb-6">
        <div className="flex flex-wrap gap-4 items-center">
          {/* Search */}
          <div className="flex-1 min-w-64">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search models..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400"
              />
              <div className="absolute inset-y-0 left-0 flex items-center pl-3">
                <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
            </div>
          </div>

          {/* Type Filter */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Type:
            </label>
            <select
              value={currentTypeFilter}
              onChange={(e) => setCurrentTypeFilter(e.target.value as typeof currentTypeFilter)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
            >
              <option value="all">All Types</option>
              {availableTypes.map(type => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Architecture Filter */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Architecture:
            </label>
            <select
              value={currentArchFilter}
              onChange={(e) => setCurrentArchFilter(e.target.value as typeof currentArchFilter)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
            >
              <option value="all">All Architectures</option>
              {availableArchitectures.map(arch => {
                const archInfo = getModelArchitectureInfo(arch);
                return (
                  <option key={arch} value={arch}>
                    {archInfo?.name || arch.toUpperCase()}
                  </option>
                );
              })}
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Sort:
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 text-sm"
            >
              <option value="created">Date Created</option>
              <option value="name">Name</option>
              <option value="type">Type</option>
              <option value="architecture">Architecture</option>
              {showPerformance && <option value="performance">Performance</option>}
            </select>
            
            <button
              onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              title={`Sort ${sortOrder === 'asc' ? 'descending' : 'ascending'}`}
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

          {/* Clear Filters */}
          {(searchQuery || currentTypeFilter !== 'all' || currentArchFilter !== 'all') && (
            <button
              onClick={() => {
                setSearchQuery('');
                setCurrentTypeFilter('all');
                setCurrentArchFilter('all');
              }}
              className="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 underline"
            >
              Clear Filters
            </button>
          )}
        </div>
      </div>

      {/* Models Grid */}
      {processedModels.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 mx-auto mb-4 text-gray-400">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            {searchQuery || currentTypeFilter !== 'all' || currentArchFilter !== 'all'
              ? 'No models match your filters'
              : 'No models available'
            }
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            {searchQuery || currentTypeFilter !== 'all' || currentArchFilter !== 'all'
              ? 'Try adjusting your search criteria or filters.'
              : 'Get started by adding your first model.'
            }
          </p>
          
          {!isReadOnly && onModelCreate && (
            <button
              onClick={handleCreateModel}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
            >
              Add Your First Model
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {processedModels.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              isSelected={selectedModelId === model.id}
              onSelect={onModelSelect}
              onDelete={onModelDelete}
              showPerformance={showPerformance}
              isReadOnly={isReadOnly}
            />
          ))}
        </div>
      )}

      {/* Summary */}
      {processedModels.length > 0 && (
        <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-750 rounded-lg">
          <div className="flex flex-wrap gap-6 text-sm text-gray-600 dark:text-gray-400">
            <div>
              <span className="font-medium">Total Models:</span> {processedModels.length}
            </div>
            <div>
              <span className="font-medium">Detection:</span> {processedModels.filter(m => m.type === 'detection').length}
            </div>
            <div>
              <span className="font-medium">Segmentation:</span> {processedModels.filter(m => m.type === 'segmentation').length}
            </div>
            {showPerformance && (
              <div>
                <span className="font-medium">Avg mAP:</span>{' '}
                {processedModels.filter(m => m.performance_metrics?.map).length > 0
                  ? (processedModels.reduce((sum, m) => sum + (m.performance_metrics?.map || 0), 0) / 
                     processedModels.filter(m => m.performance_metrics?.map).length * 100).toFixed(1) + '%'
                  : 'N/A'
                }
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};