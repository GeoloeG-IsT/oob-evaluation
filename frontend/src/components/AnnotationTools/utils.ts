// T077: Annotation Tools - Utility Functions

import { Annotation, BoundingBox, Point, Polygon, Category, ValidationResult } from './types';

// Color utilities
export const generateCategoryColors = (count: number): string[] => {
  const colors: string[] = [];
  const hueStep = 360 / count;
  
  for (let i = 0; i < count; i++) {
    const hue = (i * hueStep) % 360;
    colors.push(`hsl(${hue}, 70%, 50%)`);
  }
  
  return colors;
};

export const hexToRgba = (hex: string, alpha: number = 1): string => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return `rgba(0, 0, 0, ${alpha})`;
  
  const r = parseInt(result[1], 16);
  const g = parseInt(result[2], 16);
  const b = parseInt(result[3], 16);
  
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

// Geometry utilities
export const pointInBoundingBox = (point: Point, bbox: BoundingBox): boolean => {
  return point.x >= bbox.x &&
         point.x <= bbox.x + bbox.width &&
         point.y >= bbox.y &&
         point.y <= bbox.y + bbox.height;
};

export const pointInPolygon = (point: Point, polygon: Polygon): boolean => {
  const { x, y } = point;
  const { points } = polygon;
  let inside = false;
  
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    if (
      ((points[i].y > y) !== (points[j].y > y)) &&
      (x < (points[j].x - points[i].x) * (y - points[i].y) / (points[j].y - points[i].y) + points[i].x)
    ) {
      inside = !inside;
    }
  }
  
  return inside;
};

export const distance = (p1: Point, p2: Point): number => {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
};

export const pointInCircle = (point: Point, center: Point, radius: number): boolean => {
  return distance(point, center) <= radius;
};

export const boundingBoxFromPoints = (points: Point[]): BoundingBox => {
  if (points.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  
  const minX = Math.min(...points.map(p => p.x));
  const maxX = Math.max(...points.map(p => p.x));
  const minY = Math.min(...points.map(p => p.y));
  const maxY = Math.max(...points.map(p => p.y));
  
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  };
};

export const calculatePolygonArea = (polygon: Polygon): number => {
  const { points } = polygon;
  if (points.length < 3) return 0;
  
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length;
    area += points[i].x * points[j].y;
    area -= points[j].x * points[i].y;
  }
  
  return Math.abs(area) / 2;
};

export const calculateBoundingBoxArea = (bbox: BoundingBox): number => {
  return bbox.width * bbox.height;
};

// Coordinate transformations
export const screenToImage = (
  screenPoint: Point,
  canvasRect: DOMRect,
  imageRect: { x: number; y: number; width: number; height: number },
  scale: number,
  panX: number,
  panY: number
): Point => {
  const canvasX = screenPoint.x - canvasRect.left;
  const canvasY = screenPoint.y - canvasRect.top;
  
  const imageX = (canvasX - panX - imageRect.x) / scale;
  const imageY = (canvasY - panY - imageRect.y) / scale;
  
  return { x: imageX, y: imageY };
};

export const imageToScreen = (
  imagePoint: Point,
  imageRect: { x: number; y: number; width: number; height: number },
  scale: number,
  panX: number,
  panY: number
): Point => {
  const screenX = (imagePoint.x * scale) + imageRect.x + panX;
  const screenY = (imagePoint.y * scale) + imageRect.y + panY;
  
  return { x: screenX, y: screenY };
};

// Annotation utilities
export const createBoundingBoxAnnotation = (
  imageId: string,
  categoryId: string,
  categoryName: string,
  bbox: BoundingBox
): Omit<Annotation, 'id' | 'created_at' | 'updated_at'> => {
  return {
    image_id: imageId,
    type: 'bbox',
    category_id: categoryId,
    category_name: categoryName,
    bbox,
    area: calculateBoundingBoxArea(bbox),
    created_by: 'user',
    is_crowd: false
  };
};

export const createPolygonAnnotation = (
  imageId: string,
  categoryId: string,
  categoryName: string,
  polygon: Polygon
): Omit<Annotation, 'id' | 'created_at' | 'updated_at'> => {
  return {
    image_id: imageId,
    type: 'polygon',
    category_id: categoryId,
    category_name: categoryName,
    segmentation: polygon,
    bbox: boundingBoxFromPoints(polygon.points),
    area: calculatePolygonArea(polygon),
    created_by: 'user',
    is_crowd: false
  };
};

export const createPointAnnotation = (
  imageId: string,
  categoryId: string,
  categoryName: string,
  point: Point
): Omit<Annotation, 'id' | 'created_at' | 'updated_at'> => {
  return {
    image_id: imageId,
    type: 'point',
    category_id: categoryId,
    category_name: categoryName,
    center: point,
    area: 1,
    created_by: 'user',
    is_crowd: false
  };
};

export const createCircleAnnotation = (
  imageId: string,
  categoryId: string,
  categoryName: string,
  center: Point,
  radius: number
): Omit<Annotation, 'id' | 'created_at' | 'updated_at'> => {
  const diameter = radius * 2;
  return {
    image_id: imageId,
    type: 'circle',
    category_id: categoryId,
    category_name: categoryName,
    center,
    radius,
    bbox: {
      x: center.x - radius,
      y: center.y - radius,
      width: diameter,
      height: diameter
    },
    area: Math.PI * radius * radius,
    created_by: 'user',
    is_crowd: false
  };
};

// Hit testing
export const hitTestAnnotation = (point: Point, annotation: Annotation, tolerance: number = 5): boolean => {
  switch (annotation.type) {
    case 'bbox':
      if (!annotation.bbox) return false;
      return pointInBoundingBox(point, annotation.bbox);
      
    case 'polygon':
      if (!annotation.segmentation) return false;
      return pointInPolygon(point, annotation.segmentation);
      
    case 'point':
      if (!annotation.center) return false;
      return distance(point, annotation.center) <= tolerance;
      
    case 'circle':
      if (!annotation.center || !annotation.radius) return false;
      return pointInCircle(point, annotation.center, annotation.radius);
      
    default:
      return false;
  }
};

// Validation
export const validateAnnotation = (
  annotation: Annotation,
  imageWidth: number,
  imageHeight: number
): ValidationResult => {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  // Check if annotation is within image bounds
  switch (annotation.type) {
    case 'bbox':
      if (annotation.bbox) {
        const { x, y, width, height } = annotation.bbox;
        if (x < 0 || y < 0 || x + width > imageWidth || y + height > imageHeight) {
          errors.push(`Bounding box extends outside image bounds`);
        }
        if (width <= 0 || height <= 0) {
          errors.push(`Bounding box has invalid dimensions`);
        }
        if (width < 10 || height < 10) {
          warnings.push(`Bounding box is very small (${width}x${height})`);
        }
      }
      break;
      
    case 'polygon':
      if (annotation.segmentation) {
        const { points } = annotation.segmentation;
        if (points.length < 3) {
          errors.push(`Polygon must have at least 3 points`);
        }
        
        const outOfBounds = points.some(p => 
          p.x < 0 || p.y < 0 || p.x > imageWidth || p.y > imageHeight
        );
        if (outOfBounds) {
          errors.push(`Polygon extends outside image bounds`);
        }
        
        const area = calculatePolygonArea(annotation.segmentation);
        if (area < 100) {
          warnings.push(`Polygon area is very small (${Math.round(area)} pixels)`);
        }
      }
      break;
      
    case 'point':
      if (annotation.center) {
        const { x, y } = annotation.center;
        if (x < 0 || y < 0 || x > imageWidth || y > imageHeight) {
          errors.push(`Point is outside image bounds`);
        }
      }
      break;
      
    case 'circle':
      if (annotation.center && annotation.radius) {
        const { x, y } = annotation.center;
        const { radius } = annotation;
        if (x - radius < 0 || y - radius < 0 || 
            x + radius > imageWidth || y + radius > imageHeight) {
          errors.push(`Circle extends outside image bounds`);
        }
        if (radius < 5) {
          warnings.push(`Circle radius is very small (${radius} pixels)`);
        }
      }
      break;
  }
  
  // Check confidence
  if (annotation.confidence !== undefined && 
      (annotation.confidence < 0 || annotation.confidence > 1)) {
    errors.push(`Confidence must be between 0 and 1`);
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

export const validateAllAnnotations = (
  annotations: Annotation[],
  categories: Category[],
  imageWidth: number,
  imageHeight: number
): ValidationResult => {
  const allErrors: string[] = [];
  const allWarnings: string[] = [];
  
  const categoryIds = new Set(categories.map(c => c.id));
  
  annotations.forEach((annotation, index) => {
    // Validate category exists
    if (!categoryIds.has(annotation.category_id)) {
      allErrors.push(`Annotation ${index + 1}: Category '${annotation.category_name}' not found`);
    }
    
    // Validate annotation geometry
    const validation = validateAnnotation(annotation, imageWidth, imageHeight);
    validation.errors.forEach(error => {
      allErrors.push(`Annotation ${index + 1}: ${error}`);
    });
    validation.warnings.forEach(warning => {
      allWarnings.push(`Annotation ${index + 1}: ${warning}`);
    });
  });
  
  return {
    isValid: allErrors.length === 0,
    errors: allErrors,
    warnings: allWarnings
  };
};

// Export utilities
export const convertToCOCOFormat = (
  annotations: Annotation[],
  categories: Category[],
  imageData: { id: string; filename: string; width: number; height: number }[]
) => {
  const cocoData = {
    info: {
      description: 'ML Evaluation Platform Annotations',
      version: '1.0',
      year: new Date().getFullYear(),
      contributor: 'ML Evaluation Platform',
      date_created: new Date().toISOString()
    },
    licenses: [],
    images: imageData.map((img, index) => ({
      id: index + 1,
      width: img.width,
      height: img.height,
      file_name: img.filename,
      license: 0,
      flickr_url: '',
      coco_url: '',
      date_captured: 0
    })),
    categories: categories.map((cat, index) => ({
      id: index + 1,
      name: cat.name,
      supercategory: cat.supercategory || 'object'
    })),
    annotations: annotations.map((ann, index) => {
      const imageIndex = imageData.findIndex(img => img.id === ann.image_id);
      const categoryIndex = categories.findIndex(cat => cat.id === ann.category_id);
      
      let segmentation: number[][] = [];
      let bbox: number[] = [];
      let area = ann.area || 0;
      
      if (ann.type === 'bbox' && ann.bbox) {
        bbox = [ann.bbox.x, ann.bbox.y, ann.bbox.width, ann.bbox.height];
        area = ann.bbox.width * ann.bbox.height;
      } else if (ann.type === 'polygon' && ann.segmentation) {
        const flatPoints: number[] = [];
        ann.segmentation.points.forEach(p => {
          flatPoints.push(p.x, p.y);
        });
        segmentation = [flatPoints];
        bbox = ann.bbox ? [ann.bbox.x, ann.bbox.y, ann.bbox.width, ann.bbox.height] : [];
      }
      
      return {
        id: index + 1,
        image_id: imageIndex + 1,
        category_id: categoryIndex + 1,
        segmentation,
        area,
        bbox,
        iscrowd: ann.is_crowd ? 1 : 0
      };
    })
  };
  
  return cocoData;
};

export const convertToYOLOFormat = (
  annotations: Annotation[],
  categories: Category[],
  imageWidth: number,
  imageHeight: number
): string[] => {
  return annotations.map(ann => {
    const categoryIndex = categories.findIndex(cat => cat.id === ann.category_id);
    
    if (ann.type === 'bbox' && ann.bbox) {
      // Convert to YOLO format: class_id center_x center_y width height (normalized)
      const centerX = (ann.bbox.x + ann.bbox.width / 2) / imageWidth;
      const centerY = (ann.bbox.y + ann.bbox.height / 2) / imageHeight;
      const width = ann.bbox.width / imageWidth;
      const height = ann.bbox.height / imageHeight;
      
      return `${categoryIndex} ${centerX.toFixed(6)} ${centerY.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)}`;
    }
    
    return '';
  }).filter(line => line.length > 0);
};

// API utilities
export const createAnnotationAPI = async (annotation: Omit<Annotation, 'id' | 'created_at' | 'updated_at'>) => {
  const response = await fetch('/api/v1/annotations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotation)
  });
  
  if (!response.ok) {
    throw new Error(`Failed to create annotation: ${response.statusText}`);
  }
  
  return response.json();
};

export const updateAnnotationAPI = async (annotationId: string, updates: Partial<Annotation>) => {
  const response = await fetch(`/api/v1/annotations/${annotationId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates)
  });
  
  if (!response.ok) {
    throw new Error(`Failed to update annotation: ${response.statusText}`);
  }
  
  return response.json();
};

export const deleteAnnotationAPI = async (annotationId: string) => {
  const response = await fetch(`/api/v1/annotations/${annotationId}`, {
    method: 'DELETE'
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete annotation: ${response.statusText}`);
  }
};

export const getAssistedAnnotationsAPI = async (imageId: string, modelName: string) => {
  const response = await fetch('/api/v1/annotations/assisted', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_id: imageId,
      model_name: modelName
    })
  });
  
  if (!response.ok) {
    throw new Error(`Failed to get assisted annotations: ${response.statusText}`);
  }
  
  return response.json();
};