'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { AnnotationCanvasProps, AnnotationTool, Annotation, Point, CanvasState, BoundingBox, Polygon } from './types';
import { 
  screenToImage, 
  imageToScreen, 
  hitTestAnnotation, 
  hexToRgba, 
  createBoundingBoxAnnotation, 
  createPolygonAnnotation, 
  createPointAnnotation, 
  createCircleAnnotation,
  distance 
} from './utils';

export const AnnotationCanvas: React.FC<AnnotationCanvasProps> = ({
  imageUrl,
  imageWidth,
  imageHeight,
  annotations,
  categories,
  selectedTool,
  selectedCategoryId,
  onAnnotationCreate,
  onAnnotationUpdate,
  onAnnotationDelete,
  onAnnotationSelect,
  selectedAnnotationId,
  isReadOnly = false,
  showLabels = true,
  showConfidence = false,
  minConfidence = 0,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  
  const [canvasState, setCanvasState] = useState<CanvasState>({
    scale: 1,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragStart: null,
    isDrawing: false,
    currentAnnotation: null,
    selectedAnnotationId: null,
    hoveredAnnotationId: null,
  });

  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageRect, setImageRect] = useState({ x: 0, y: 0, width: 0, height: 0 });

  // Initialize image
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setImageLoaded(true);
      imageRef.current = img;
      fitImageToCanvas();
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Fit image to canvas
  const fitImageToCanvas = useCallback(() => {
    if (!canvasRef.current || !containerRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = imageRef.current;

    // Set canvas size to container
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Calculate scale to fit image in canvas
    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;
    const scale = Math.min(scaleX, scaleY, 1); // Don't scale up

    // Center the image
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    const x = (canvas.width - scaledWidth) / 2;
    const y = (canvas.height - scaledHeight) / 2;

    setImageRect({ x, y, width: scaledWidth, height: scaledHeight });
    setCanvasState(prev => ({
      ...prev,
      scale,
      panX: 0,
      panY: 0,
    }));
  }, []);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => fitImageToCanvas();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [fitImageToCanvas]);

  // Draw everything on canvas
  const draw = useCallback(() => {
    if (!canvasRef.current || !imageRef.current || !imageLoaded) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Save context
    ctx.save();

    // Apply pan and zoom
    ctx.translate(canvasState.panX, canvasState.panY);

    // Draw image
    ctx.drawImage(
      imageRef.current,
      imageRect.x,
      imageRect.y,
      imageRect.width,
      imageRect.height
    );

    // Filter annotations by confidence
    const visibleAnnotations = annotations.filter(ann => 
      !ann.confidence || ann.confidence >= minConfidence
    );

    // Draw annotations
    visibleAnnotations.forEach(annotation => {
      const category = categories.find(cat => cat.id === annotation.category_id);
      const color = category?.color || '#ff0000';
      const isSelected = annotation.id === selectedAnnotationId;
      const isHovered = annotation.id === canvasState.hoveredAnnotationId;
      
      drawAnnotation(ctx, annotation, color, isSelected, isHovered);
    });

    // Draw current annotation being created
    if (canvasState.currentAnnotation && canvasState.isDrawing) {
      const category = categories.find(cat => cat.id === selectedCategoryId);
      const color = category?.color || '#ff0000';
      drawAnnotation(ctx, canvasState.currentAnnotation as Annotation, color, false, false, true);
    }

    // Restore context
    ctx.restore();
  }, [
    canvasState, 
    imageLoaded, 
    imageRect, 
    annotations, 
    categories, 
    selectedAnnotationId, 
    selectedCategoryId, 
    minConfidence
  ]);

  // Draw individual annotation
  const drawAnnotation = useCallback((
    ctx: CanvasRenderingContext2D,
    annotation: Annotation,
    color: string,
    isSelected: boolean,
    isHovered: boolean,
    isPreview: boolean = false
  ) => {
    const alpha = isPreview ? 0.5 : (isSelected ? 0.8 : 0.6);
    const strokeWidth = isSelected ? 3 : (isHovered ? 2 : 1);
    
    ctx.strokeStyle = color;
    ctx.fillStyle = hexToRgba(color, 0.2);
    ctx.lineWidth = strokeWidth;

    switch (annotation.type) {
      case 'bbox':
        if (annotation.bbox) {
          const screenBBox = {
            x: (annotation.bbox.x * canvasState.scale) + imageRect.x,
            y: (annotation.bbox.y * canvasState.scale) + imageRect.y,
            width: annotation.bbox.width * canvasState.scale,
            height: annotation.bbox.height * canvasState.scale
          };
          
          ctx.fillRect(screenBBox.x, screenBBox.y, screenBBox.width, screenBBox.height);
          ctx.strokeRect(screenBBox.x, screenBBox.y, screenBBox.width, screenBBox.height);
          
          // Draw label
          if (showLabels && !isPreview) {
            drawLabel(ctx, annotation, { x: screenBBox.x, y: screenBBox.y - 5 }, color);
          }
        }
        break;

      case 'polygon':
        if (annotation.segmentation && annotation.segmentation.points.length > 0) {
          ctx.beginPath();
          annotation.segmentation.points.forEach((point, index) => {
            const screenPoint = imageToScreen(point, imageRect, canvasState.scale, 0, 0);
            if (index === 0) {
              ctx.moveTo(screenPoint.x, screenPoint.y);
            } else {
              ctx.lineTo(screenPoint.x, screenPoint.y);
            }
          });
          
          if (annotation.segmentation.closed !== false) {
            ctx.closePath();
          }
          
          ctx.fill();
          ctx.stroke();
          
          // Draw points
          annotation.segmentation.points.forEach(point => {
            const screenPoint = imageToScreen(point, imageRect, canvasState.scale, 0, 0);
            ctx.beginPath();
            ctx.arc(screenPoint.x, screenPoint.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
          });
          
          // Draw label
          if (showLabels && !isPreview && annotation.segmentation.points.length > 0) {
            const firstPoint = imageToScreen(annotation.segmentation.points[0], imageRect, canvasState.scale, 0, 0);
            drawLabel(ctx, annotation, { x: firstPoint.x, y: firstPoint.y - 10 }, color);
          }
        }
        break;

      case 'point':
        if (annotation.center) {
          const screenPoint = imageToScreen(annotation.center, imageRect, canvasState.scale, 0, 0);
          ctx.beginPath();
          ctx.arc(screenPoint.x, screenPoint.y, 8, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.stroke();
          
          // Draw label
          if (showLabels && !isPreview) {
            drawLabel(ctx, annotation, { x: screenPoint.x + 10, y: screenPoint.y - 10 }, color);
          }
        }
        break;

      case 'circle':
        if (annotation.center && annotation.radius) {
          const screenCenter = imageToScreen(annotation.center, imageRect, canvasState.scale, 0, 0);
          const screenRadius = annotation.radius * canvasState.scale;
          
          ctx.beginPath();
          ctx.arc(screenCenter.x, screenCenter.y, screenRadius, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
          
          // Draw center point
          ctx.beginPath();
          ctx.arc(screenCenter.x, screenCenter.y, 3, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          
          // Draw label
          if (showLabels && !isPreview) {
            drawLabel(ctx, annotation, { x: screenCenter.x + screenRadius + 5, y: screenCenter.y - 10 }, color);
          }
        }
        break;
    }
  }, [canvasState, imageRect, showLabels]);

  // Draw label
  const drawLabel = useCallback((
    ctx: CanvasRenderingContext2D,
    annotation: Annotation,
    position: Point,
    color: string
  ) => {
    let labelText = annotation.category_name;
    
    if (showConfidence && annotation.confidence !== undefined) {
      labelText += ` (${(annotation.confidence * 100).toFixed(1)}%)`;
    }
    
    ctx.font = '12px Arial';
    const metrics = ctx.measureText(labelText);
    const padding = 4;
    
    // Background
    ctx.fillStyle = color;
    ctx.fillRect(
      position.x - padding,
      position.y - 12 - padding,
      metrics.width + padding * 2,
      12 + padding * 2
    );
    
    // Text
    ctx.fillStyle = '#fff';
    ctx.fillText(labelText, position.x, position.y);
  }, [showConfidence]);

  // Mouse event handlers
  const getMousePosition = useCallback((e: React.MouseEvent): Point => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (isReadOnly) return;
    
    const mousePos = getMousePosition(e);
    const imagePos = screenToImage(mousePos, canvasRef.current!.getBoundingClientRect(), imageRect, canvasState.scale, canvasState.panX, canvasState.panY);
    
    if (selectedTool === 'pan') {
      setCanvasState(prev => ({
        ...prev,
        isDragging: true,
        dragStart: mousePos,
      }));
      return;
    }
    
    if (selectedTool === 'select') {
      // Find clicked annotation
      const clickedAnnotation = annotations.find(ann => 
        hitTestAnnotation(imagePos, ann)
      );
      
      onAnnotationSelect?.(clickedAnnotation || null);
      setCanvasState(prev => ({ 
        ...prev, 
        selectedAnnotationId: clickedAnnotation?.id || null 
      }));
      return;
    }
    
    // Start creating new annotation
    const category = categories.find(cat => cat.id === selectedCategoryId);
    if (!category) return;
    
    setCanvasState(prev => ({
      ...prev,
      isDrawing: true,
      dragStart: imagePos,
    }));
    
    if (selectedTool === 'point') {
      // Create point annotation immediately
      const annotation = createPointAnnotation('current-image', selectedCategoryId, category.name, imagePos);
      onAnnotationCreate?.(annotation);
      setCanvasState(prev => ({
        ...prev,
        isDrawing: false,
        currentAnnotation: null,
      }));
    } else {
      // Start creating other annotation types
      let currentAnnotation: Partial<Annotation> | null = null;
      
      switch (selectedTool) {
        case 'bbox':
          currentAnnotation = {
            image_id: 'current-image',
            type: 'bbox',
            category_id: selectedCategoryId,
            category_name: category.name,
            bbox: { x: imagePos.x, y: imagePos.y, width: 0, height: 0 },
            created_by: 'user',
          };
          break;
          
        case 'polygon':
          currentAnnotation = {
            image_id: 'current-image',
            type: 'polygon',
            category_id: selectedCategoryId,
            category_name: category.name,
            segmentation: { points: [imagePos], closed: false },
            created_by: 'user',
          };
          break;
          
        case 'circle':
          currentAnnotation = {
            image_id: 'current-image',
            type: 'circle',
            category_id: selectedCategoryId,
            category_name: category.name,
            center: imagePos,
            radius: 0,
            created_by: 'user',
          };
          break;
      }
      
      setCanvasState(prev => ({
        ...prev,
        currentAnnotation,
      }));
    }
  }, [
    isReadOnly, selectedTool, categories, selectedCategoryId, annotations,
    onAnnotationCreate, onAnnotationSelect, getMousePosition, canvasState, imageRect
  ]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const mousePos = getMousePosition(e);
    const imagePos = screenToImage(mousePos, canvasRef.current!.getBoundingClientRect(), imageRect, canvasState.scale, canvasState.panX, canvasState.panY);
    
    if (canvasState.isDragging && selectedTool === 'pan' && canvasState.dragStart) {
      // Pan the canvas
      const deltaX = mousePos.x - canvasState.dragStart.x;
      const deltaY = mousePos.y - canvasState.dragStart.y;
      
      setCanvasState(prev => ({
        ...prev,
        panX: prev.panX + deltaX,
        panY: prev.panY + deltaY,
        dragStart: mousePos,
      }));
      return;
    }
    
    if (canvasState.isDrawing && canvasState.dragStart && canvasState.currentAnnotation) {
      // Update current annotation being drawn
      const updatedAnnotation = { ...canvasState.currentAnnotation };
      
      switch (selectedTool) {
        case 'bbox':
          if (updatedAnnotation.bbox) {
            updatedAnnotation.bbox = {
              x: Math.min(canvasState.dragStart.x, imagePos.x),
              y: Math.min(canvasState.dragStart.y, imagePos.y),
              width: Math.abs(imagePos.x - canvasState.dragStart.x),
              height: Math.abs(imagePos.y - canvasState.dragStart.y),
            };
          }
          break;
          
        case 'circle':
          if (updatedAnnotation.center) {
            updatedAnnotation.radius = distance(updatedAnnotation.center, imagePos);
          }
          break;
      }
      
      setCanvasState(prev => ({
        ...prev,
        currentAnnotation: updatedAnnotation,
      }));
    }
    
    // Update hovered annotation for visual feedback
    if (!canvasState.isDrawing && selectedTool === 'select') {
      const hoveredAnnotation = annotations.find(ann => 
        hitTestAnnotation(imagePos, ann)
      );
      
      setCanvasState(prev => ({
        ...prev,
        hoveredAnnotationId: hoveredAnnotation?.id || null,
      }));
    }
  }, [
    getMousePosition, canvasState, selectedTool, imageRect, annotations
  ]);

  const handleMouseUp = useCallback(() => {
    if (canvasState.isDrawing && canvasState.currentAnnotation) {
      // Finish creating annotation
      const annotation = canvasState.currentAnnotation;
      
      // Validate annotation has minimum size/content
      let isValid = false;
      switch (selectedTool) {
        case 'bbox':
          isValid = annotation.bbox && annotation.bbox.width > 5 && annotation.bbox.height > 5;
          break;
        case 'circle':
          isValid = annotation.radius && annotation.radius > 5;
          break;
        case 'polygon':
          isValid = annotation.segmentation && annotation.segmentation.points.length >= 3;
          break;
        default:
          isValid = true;
      }
      
      if (isValid) {
        onAnnotationCreate?.(annotation as Omit<Annotation, 'id' | 'created_at' | 'updated_at'>);
      }
    }
    
    setCanvasState(prev => ({
      ...prev,
      isDragging: false,
      isDrawing: false,
      dragStart: null,
      currentAnnotation: null,
    }));
  }, [canvasState, selectedTool, onAnnotationCreate]);

  // Double-click handler for polygon completion
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    if (selectedTool === 'polygon' && canvasState.isDrawing && canvasState.currentAnnotation) {
      const annotation = { ...canvasState.currentAnnotation };
      if (annotation.segmentation && annotation.segmentation.points.length >= 3) {
        annotation.segmentation.closed = true;
        onAnnotationCreate?.(annotation as Omit<Annotation, 'id' | 'created_at' | 'updated_at'>);
        
        setCanvasState(prev => ({
          ...prev,
          isDrawing: false,
          currentAnnotation: null,
        }));
      }
    }
  }, [selectedTool, canvasState, onAnnotationCreate]);

  // Single click handler for polygon points
  const handleClick = useCallback((e: React.MouseEvent) => {
    if (selectedTool === 'polygon' && canvasState.isDrawing && canvasState.currentAnnotation) {
      const mousePos = getMousePosition(e);
      const imagePos = screenToImage(mousePos, canvasRef.current!.getBoundingClientRect(), imageRect, canvasState.scale, canvasState.panX, canvasState.panY);
      
      const annotation = { ...canvasState.currentAnnotation };
      if (annotation.segmentation) {
        annotation.segmentation.points.push(imagePos);
        setCanvasState(prev => ({
          ...prev,
          currentAnnotation: annotation,
        }));
      }
    }
  }, [selectedTool, canvasState, getMousePosition, imageRect]);

  // Keyboard handlers
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && canvasState.isDrawing) {
        setCanvasState(prev => ({
          ...prev,
          isDrawing: false,
          currentAnnotation: null,
        }));
      }
      
      if (e.key === 'Delete' && selectedAnnotationId) {
        onAnnotationDelete?.(selectedAnnotationId);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [canvasState.isDrawing, selectedAnnotationId, onAnnotationDelete]);

  // Redraw when state changes
  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div 
      ref={containerRef}
      className={`relative w-full h-full overflow-hidden bg-gray-100 dark:bg-gray-800 ${className}`}
    >
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        className={`cursor-${
          selectedTool === 'pan' ? 'grab' :
          selectedTool === 'select' ? 'pointer' :
          'crosshair'
        } w-full h-full`}
      />
      
      {/* Loading overlay */}
      {!imageLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
            <p className="text-gray-600 dark:text-gray-400">Loading image...</p>
          </div>
        </div>
      )}
    </div>
  );
};