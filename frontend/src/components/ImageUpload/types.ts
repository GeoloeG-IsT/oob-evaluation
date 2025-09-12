// T076: Image Upload and Gallery - Type Definitions

export interface UploadedImage {
  id: string;
  filename: string;
  original_filename: string;
  file_path: string;
  file_size: number;
  format: string;
  width?: number;
  height?: number;
  channels?: number;
  color_mode?: string;
  dataset_split: 'train' | 'val' | 'test';
  upload_date: string;
  metadata?: Record<string, unknown>;
  thumbnail_url?: string;
  preview_url?: string;
}

export interface UploadProgress {
  filename: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
}

export interface ImageUploadProps {
  onUploadComplete?: (images: UploadedImage[]) => void;
  onUploadProgress?: (progress: UploadProgress[]) => void;
  acceptedFormats?: string[];
  maxFileSize?: number; // in bytes
  allowMultiple?: boolean;
  defaultDatasetSplit?: 'train' | 'val' | 'test';
  className?: string;
}

export interface ImageGalleryProps {
  images: UploadedImage[];
  onImageSelect?: (image: UploadedImage) => void;
  onImageDelete?: (imageId: string) => void;
  selectedImages?: string[];
  allowMultipleSelect?: boolean;
  showMetadata?: boolean;
  filterByDatasetSplit?: 'train' | 'val' | 'test' | 'all';
  sortBy?: 'upload_date' | 'filename' | 'file_size';
  sortOrder?: 'asc' | 'desc';
  className?: string;
}

export interface ImageCardProps {
  image: UploadedImage;
  isSelected?: boolean;
  onSelect?: (image: UploadedImage) => void;
  onDelete?: (imageId: string) => void;
  showMetadata?: boolean;
  className?: string;
}

export interface DatasetSplitManagerProps {
  images: UploadedImage[];
  onSplitChange?: (imageId: string, newSplit: 'train' | 'val' | 'test') => void;
  splitDistribution?: {
    train: number;
    val: number;
    test: number;
  };
  className?: string;
}

export interface ImagePreviewModalProps {
  image: UploadedImage | null;
  isOpen: boolean;
  onClose: () => void;
  onNavigate?: (direction: 'prev' | 'next') => void;
  showMetadata?: boolean;
}

export interface UploadApiResponse {
  success: boolean;
  images?: UploadedImage[];
  errors?: string[];
  message?: string;
}

export interface ImageFiltersProps {
  currentFilter: {
    datasetSplit: 'train' | 'val' | 'test' | 'all';
    sortBy: 'upload_date' | 'filename' | 'file_size';
    sortOrder: 'asc' | 'desc';
  };
  onFilterChange: (filter: Partial<ImageFiltersProps['currentFilter']>) => void;
  totalCounts: {
    all: number;
    train: number;
    val: number;
    test: number;
  };
  className?: string;
}