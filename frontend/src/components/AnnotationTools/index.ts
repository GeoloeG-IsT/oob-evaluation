// T077: Annotation Tools - Main Export

export { AnnotationCanvas } from './AnnotationCanvas';
export { AnnotationToolbar } from './AnnotationToolbar';
export { CategoryManager } from './CategoryManager';
export { AnnotationList } from './AnnotationList';
export { AssistedAnnotation } from './AssistedAnnotation';

export type * from './types';

export {
  generateCategoryColors,
  hexToRgba,
  pointInBoundingBox,
  pointInPolygon,
  distance,
  pointInCircle,
  boundingBoxFromPoints,
  calculatePolygonArea,
  calculateBoundingBoxArea,
  screenToImage,
  imageToScreen,
  createBoundingBoxAnnotation,
  createPolygonAnnotation,
  createPointAnnotation,
  createCircleAnnotation,
  hitTestAnnotation,
  validateAnnotation,
  validateAllAnnotations,
  convertToCOCOFormat,
  convertToYOLOFormat,
  createAnnotationAPI,
  updateAnnotationAPI,
  deleteAnnotationAPI,
  getAssistedAnnotationsAPI,
} from './utils';