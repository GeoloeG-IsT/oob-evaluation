'use client';

import React, { useState } from 'react';
import { CategoryManagerProps, Category } from './types';
import { generateCategoryColors } from './utils';

export const CategoryManager: React.FC<CategoryManagerProps> = ({
  categories,
  selectedCategoryId,
  onCategorySelect,
  onCategoryCreate,
  onCategoryUpdate,
  onCategoryDelete,
  isReadOnly = false,
  className = '',
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newCategory, setNewCategory] = useState({
    name: '',
    supercategory: '',
    color: '#ff0000',
    description: '',
  });

  const handleCreateCategory = () => {
    if (!newCategory.name.trim()) return;

    const categoryWithColor = {
      ...newCategory,
      color: newCategory.color || generateCategoryColors(1)[0],
    };

    onCategoryCreate?.(categoryWithColor);
    setNewCategory({
      name: '',
      supercategory: '',
      color: '#ff0000',
      description: '',
    });
    setIsCreating(false);
  };

  const handleUpdateCategory = (categoryId: string, updates: Partial<Category>) => {
    onCategoryUpdate?.(categoryId, updates);
    setEditingId(null);
  };

  const handleDeleteCategory = (categoryId: string) => {
    if (confirm('Are you sure you want to delete this category? This cannot be undone.')) {
      onCategoryDelete?.(categoryId);
    }
  };

  const getRandomColor = () => {
    const colors = [
      '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
      '#ff8000', '#8000ff', '#0080ff', '#80ff00', '#ff0080', '#0040ff',
      '#40ff00', '#ff4000', '#8040ff', '#4080ff', '#80ff40', '#ff8040'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  return (
    <div className={`category-manager bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Categories ({categories.length})
        </h3>
        {!isReadOnly && onCategoryCreate && (
          <button
            onClick={() => setIsCreating(!isCreating)}
            className={`px-3 py-2 text-sm rounded transition-colors ${
              isCreating
                ? 'bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            {isCreating ? 'Cancel' : 'Add Category'}
          </button>
        )}
      </div>

      {/* Create New Category Form */}
      {isCreating && (
        <div className="p-4 bg-gray-50 dark:bg-gray-750 border-b border-gray-200 dark:border-gray-700">
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Name *
              </label>
              <input
                type="text"
                value={newCategory.name}
                onChange={(e) => setNewCategory(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Enter category name"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Supercategory
              </label>
              <input
                type="text"
                value={newCategory.supercategory}
                onChange={(e) => setNewCategory(prev => ({ ...prev, supercategory: e.target.value }))}
                placeholder="e.g., vehicle, animal, person"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Color
              </label>
              <div className="flex items-center space-x-2">
                <input
                  type="color"
                  value={newCategory.color}
                  onChange={(e) => setNewCategory(prev => ({ ...prev, color: e.target.value }))}
                  className="w-12 h-8 border border-gray-300 dark:border-gray-600 rounded cursor-pointer"
                />
                <button
                  onClick={() => setNewCategory(prev => ({ ...prev, color: getRandomColor() }))}
                  className="px-2 py-1 text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-500"
                >
                  Random
                </button>
                <span className="text-sm text-gray-600 dark:text-gray-400">{newCategory.color}</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Description
              </label>
              <textarea
                value={newCategory.description}
                onChange={(e) => setNewCategory(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Optional description"
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm"
              />
            </div>

            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setIsCreating(false)}
                className="px-3 py-2 text-sm bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-500"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateCategory}
                disabled={!newCategory.name.trim()}
                className="px-3 py-2 text-sm bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 dark:disabled:bg-blue-700 text-white rounded disabled:cursor-not-allowed"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Categories List */}
      <div className="max-h-96 overflow-y-auto">
        {categories.length === 0 ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            <div className="w-12 h-12 mx-auto mb-3 text-gray-400">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
            </div>
            <p className="text-lg font-medium mb-1">No Categories</p>
            <p className="text-sm">Create categories to start annotating images.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {categories.map((category) => (
              <div
                key={category.id}
                className={`p-4 transition-colors cursor-pointer ${
                  selectedCategoryId === category.id
                    ? 'bg-blue-50 dark:bg-blue-950'
                    : 'hover:bg-gray-50 dark:hover:bg-gray-750'
                }`}
                onClick={() => onCategorySelect(category.id)}
              >
                {editingId === category.id ? (
                  /* Edit Mode */
                  <EditCategoryForm
                    category={category}
                    onSave={(updates) => handleUpdateCategory(category.id, updates)}
                    onCancel={() => setEditingId(null)}
                  />
                ) : (
                  /* Display Mode */
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3 flex-1">
                      {/* Color Swatch */}
                      <div
                        className="w-6 h-6 rounded border-2 border-white shadow-sm flex-shrink-0"
                        style={{ backgroundColor: category.color }}
                      />
                      
                      {/* Category Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <h4 className="font-medium text-gray-900 dark:text-gray-100 truncate">
                            {category.name}
                          </h4>
                          {selectedCategoryId === category.id && (
                            <span className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded">
                              Selected
                            </span>
                          )}
                        </div>
                        {category.supercategory && (
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {category.supercategory}
                          </p>
                        )}
                        {category.description && (
                          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 truncate">
                            {category.description}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Actions */}
                    {!isReadOnly && (
                      <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        {onCategoryUpdate && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingId(category.id);
                            }}
                            className="p-1 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400"
                            title="Edit category"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                          </button>
                        )}
                        {onCategoryDelete && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteCategory(category.id);
                            }}
                            className="p-1 text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                            title="Delete category"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Edit Category Form Component
const EditCategoryForm: React.FC<{
  category: Category;
  onSave: (updates: Partial<Category>) => void;
  onCancel: () => void;
}> = ({ category, onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    name: category.name,
    supercategory: category.supercategory || '',
    color: category.color,
    description: category.description || '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name.trim()) return;
    onSave(formData);
  };

  const getRandomColor = () => {
    const colors = [
      '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
      '#ff8000', '#8000ff', '#0080ff', '#80ff00', '#ff0080', '#0040ff'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
          placeholder="Category name"
          className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        />
      </div>

      <div>
        <input
          type="text"
          value={formData.supercategory}
          onChange={(e) => setFormData(prev => ({ ...prev, supercategory: e.target.value }))}
          placeholder="Supercategory (optional)"
          className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        />
      </div>

      <div className="flex items-center space-x-2">
        <input
          type="color"
          value={formData.color}
          onChange={(e) => setFormData(prev => ({ ...prev, color: e.target.value }))}
          className="w-8 h-6 border border-gray-300 dark:border-gray-600 rounded cursor-pointer"
        />
        <button
          type="button"
          onClick={() => setFormData(prev => ({ ...prev, color: getRandomColor() }))}
          className="px-2 py-1 text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-500"
        >
          Random
        </button>
      </div>

      <div>
        <textarea
          value={formData.description}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Description (optional)"
          rows={2}
          className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        />
      </div>

      <div className="flex justify-end space-x-2">
        <button
          type="button"
          onClick={onCancel}
          className="px-2 py-1 text-sm bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-500"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!formData.name.trim()}
          className="px-2 py-1 text-sm bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 dark:disabled:bg-blue-700 text-white rounded disabled:cursor-not-allowed"
        >
          Save
        </button>
      </div>
    </form>
  );
};