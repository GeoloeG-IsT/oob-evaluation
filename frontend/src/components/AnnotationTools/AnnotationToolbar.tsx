'use client';

import React from 'react';
import { AnnotationToolbarProps, AnnotationTool } from './types';

export const AnnotationToolbar: React.FC<AnnotationToolbarProps> = ({
  selectedTool,
  onToolSelect,
  categories,
  selectedCategoryId,
  onCategorySelect,
  canUndo = false,
  canRedo = false,
  onUndo,
  onRedo,
  onClear,
  onSave,
  onLoad,
  isReadOnly = false,
  className = '',
}) => {
  const tools: { id: AnnotationTool; name: string; icon: string; description: string }[] = [
    {
      id: 'select',
      name: 'Select',
      icon: '↖',
      description: 'Select and edit annotations',
    },
    {
      id: 'bbox',
      name: 'Bounding Box',
      icon: '▢',
      description: 'Draw bounding boxes',
    },
    {
      id: 'polygon',
      name: 'Polygon',
      icon: '⬟',
      description: 'Draw polygons for segmentation',
    },
    {
      id: 'point',
      name: 'Point',
      icon: '•',
      description: 'Place point markers',
    },
    {
      id: 'circle',
      name: 'Circle',
      icon: '○',
      description: 'Draw circles',
    },
    {
      id: 'pan',
      name: 'Pan',
      icon: '✋',
      description: 'Pan around the image',
    },
  ];

  return (
    <div className={`annotation-toolbar bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 ${className}`}>
      <div className="flex flex-wrap items-center gap-2 p-4">
        {/* Drawing Tools */}
        <div className="flex items-center space-x-1 mr-4">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300 mr-2">
            Tools:
          </span>
          {tools.map((tool) => (
            <button
              key={tool.id}
              onClick={() => onToolSelect(tool.id)}
              disabled={isReadOnly && tool.id !== 'select' && tool.id !== 'pan'}
              className={`
                flex items-center justify-center w-10 h-10 rounded-lg transition-colors
                ${selectedTool === tool.id
                  ? 'bg-blue-500 text-white shadow-md'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }
                ${isReadOnly && tool.id !== 'select' && tool.id !== 'pan'
                  ? 'opacity-50 cursor-not-allowed'
                  : 'cursor-pointer'
                }
              `}
              title={tool.description}
            >
              <span className="text-lg">{tool.icon}</span>
            </button>
          ))}
        </div>

        {/* Category Selection */}
        <div className="flex items-center space-x-2 mr-4">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Category:
          </span>
          <select
            value={selectedCategoryId}
            onChange={(e) => onCategorySelect(e.target.value)}
            disabled={isReadOnly}
            className={`
              px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded
              bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
              ${isReadOnly ? 'opacity-50 cursor-not-allowed' : 'hover:border-gray-400 dark:hover:border-gray-500'}
            `}
          >
            {categories.map((category) => (
              <option key={category.id} value={category.id}>
                {category.name}
              </option>
            ))}
          </select>
          
          {/* Category Color Indicator */}
          {selectedCategoryId && (
            <div
              className="w-6 h-6 rounded border-2 border-white shadow-sm"
              style={{
                backgroundColor: categories.find(c => c.id === selectedCategoryId)?.color || '#ff0000'
              }}
              title={categories.find(c => c.id === selectedCategoryId)?.name}
            />
          )}
        </div>

        {/* History Controls */}
        <div className="flex items-center space-x-1 mr-4">
          <button
            onClick={onUndo}
            disabled={!canUndo || isReadOnly}
            className={`
              flex items-center justify-center w-8 h-8 rounded transition-colors
              ${canUndo && !isReadOnly
                ? 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                : 'bg-gray-50 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed'
              }
            `}
            title="Undo (Ctrl+Z)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
            </svg>
          </button>
          
          <button
            onClick={onRedo}
            disabled={!canRedo || isReadOnly}
            className={`
              flex items-center justify-center w-8 h-8 rounded transition-colors
              ${canRedo && !isReadOnly
                ? 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                : 'bg-gray-50 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed'
              }
            `}
            title="Redo (Ctrl+Y)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2m18-10l-6 6m6-6l-6-6" />
            </svg>
          </button>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          {onClear && (
            <button
              onClick={onClear}
              disabled={isReadOnly}
              className={`
                px-3 py-2 text-sm rounded transition-colors
                ${isReadOnly
                  ? 'bg-gray-50 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed'
                  : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200 hover:bg-red-200 dark:hover:bg-red-800'
                }
              `}
              title="Clear all annotations"
            >
              Clear All
            </button>
          )}

          {onSave && (
            <button
              onClick={onSave}
              disabled={isReadOnly}
              className={`
                px-3 py-2 text-sm rounded transition-colors
                ${isReadOnly
                  ? 'bg-gray-50 dark:bg-gray-800 text-gray-400 dark:text-gray-600 cursor-not-allowed'
                  : 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200 hover:bg-blue-200 dark:hover:bg-blue-800'
                }
              `}
              title="Save annotations"
            >
              Save
            </button>
          )}

          {onLoad && (
            <button
              onClick={onLoad}
              className="px-3 py-2 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              title="Load annotations"
            >
              Load
            </button>
          )}
        </div>
      </div>

      {/* Tool Instructions */}
      <div className="px-4 pb-3">
        <div className="text-xs text-gray-600 dark:text-gray-400">
          {selectedTool === 'select' && 'Click on annotations to select them. Selected annotations can be moved or deleted.'}
          {selectedTool === 'bbox' && 'Click and drag to create bounding boxes around objects.'}
          {selectedTool === 'polygon' && 'Click to add points, double-click to complete polygon. Press Escape to cancel.'}
          {selectedTool === 'point' && 'Click to place point annotations.'}
          {selectedTool === 'circle' && 'Click and drag from center to create circles.'}
          {selectedTool === 'pan' && 'Click and drag to pan around the image.'}
        </div>
      </div>

      {/* Keyboard Shortcuts Help */}
      <details className="px-4 pb-3">
        <summary className="text-xs text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700 dark:hover:text-gray-200">
          Keyboard Shortcuts
        </summary>
        <div className="mt-2 text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">S</kbd> - Select tool
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">B</kbd> - Bounding box
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">P</kbd> - Polygon
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">.</kbd> - Point
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">C</kbd> - Circle
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">H</kbd> - Pan (Hand)
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">Ctrl+Z</kbd> - Undo
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">Ctrl+Y</kbd> - Redo
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">Del</kbd> - Delete selected
            </div>
            <div>
              <kbd className="px-1 bg-gray-100 dark:bg-gray-800 rounded">Esc</kbd> - Cancel drawing
            </div>
          </div>
        </div>
      </details>
    </div>
  );
};