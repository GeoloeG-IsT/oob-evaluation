// T077: Annotation Tools - Type Definitions

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface Polygon {
  points: Point[];
  closed?: boolean;
}

export interface Annotation {
  id: string;
  image_id: string;
  type: 'bbox' | 'polygon' | 'point' | 'circle';
  category_id: string;
  category_name: string;
  bbox?: BoundingBox;
  segmentation?: Polygon;
  center?: Point;
  radius?: number;
  area?: number;
  confidence?: number;
  is_crowd?: boolean;
  created_by: 'user' | 'model';
  model_name?: string;
  model_version?: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

export interface Category {
  id: string;
  name: string;
  supercategory?: string;
  color: string;
  description?: string;
}

export interface AnnotationCanvasProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  annotations: Annotation[];
  categories: Category[];
  selectedTool: AnnotationTool;
  selectedCategoryId: string;
  onAnnotationCreate?: (annotation: Omit<Annotation, 'id' | 'created_at' | 'updated_at'>) => void;
  onAnnotationUpdate?: (annotationId: string, updates: Partial<Annotation>) => void;
  onAnnotationDelete?: (annotationId: string) => void;
  onAnnotationSelect?: (annotation: Annotation | null) => void;
  selectedAnnotationId?: string;
  isReadOnly?: boolean;
  showLabels?: boolean;
  showConfidence?: boolean;
  minConfidence?: number;
  className?: string;
}

export interface AnnotationToolbarProps {
  selectedTool: AnnotationTool;
  onToolSelect: (tool: AnnotationTool) => void;
  categories: Category[];
  selectedCategoryId: string;
  onCategorySelect: (categoryId: string) => void;
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo?: () => void;
  onRedo?: () => void;
  onClear?: () => void;
  onSave?: () => void;
  onLoad?: () => void;
  isReadOnly?: boolean;
  className?: string;
}

export interface CategoryManagerProps {
  categories: Category[];
  selectedCategoryId: string;
  onCategorySelect: (categoryId: string) => void;
  onCategoryCreate?: (category: Omit<Category, 'id'>) => void;
  onCategoryUpdate?: (categoryId: string, updates: Partial<Category>) => void;
  onCategoryDelete?: (categoryId: string) => void;
  isReadOnly?: boolean;
  className?: string;
}

export interface AnnotationListProps {
  annotations: Annotation[];
  categories: Category[];
  onAnnotationSelect?: (annotation: Annotation) => void;
  onAnnotationDelete?: (annotationId: string) => void;
  onAnnotationUpdate?: (annotationId: string, updates: Partial<Annotation>) => void;
  selectedAnnotationId?: string;
  showConfidence?: boolean;
  className?: string;
}

export interface AssistedAnnotationProps {
  imageUrl: string;
  onAnnotationsGenerated?: (annotations: Annotation[]) => void;
  onError?: (error: string) => void;
  models: AssistanceModel[];
  selectedModelId?: string;
  onModelSelect?: (modelId: string) => void;
  className?: string;
}

export interface AssistanceModel {
  id: string;
  name: string;
  type: 'detection' | 'segmentation' | 'sam';
  description?: string;
  version?: string;
  supported_formats: string[];
  capabilities: {
    bbox?: boolean;
    polygon?: boolean;
    point?: boolean;
    interactive?: boolean;
  };
}

export type AnnotationTool = 
  | 'select'
  | 'bbox' 
  | 'polygon'
  | 'point'
  | 'circle'
  | 'pan'
  | 'zoom';

export interface CanvasState {
  scale: number;
  panX: number;
  panY: number;
  isDragging: boolean;
  dragStart: Point | null;
  isDrawing: boolean;
  currentAnnotation: Partial<Annotation> | null;
  selectedAnnotationId: string | null;
  hoveredAnnotationId: string | null;
}

export interface AnnotationHistory {
  past: Annotation[][];
  present: Annotation[];
  future: Annotation[][];
}

export interface AnnotationStatsProps {
  annotations: Annotation[];
  categories: Category[];
  className?: string;
}

export interface ExportOptions {
  format: 'coco' | 'yolo' | 'pascal_voc' | 'csv';
  includeImages?: boolean;
  categoryFilter?: string[];
  confidenceThreshold?: number;
}

export interface AnnotationExportProps {
  annotations: Annotation[];
  categories: Category[];
  imageData?: {
    id: string;
    filename: string;
    width: number;
    height: number;
  }[];
  onExport?: (format: ExportOptions['format'], data: unknown) => void;
  className?: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

export interface AnnotationValidationProps {
  annotations: Annotation[];
  categories: Category[];
  imageWidth: number;
  imageHeight: number;
  onValidationComplete?: (result: ValidationResult) => void;
  className?: string;
}