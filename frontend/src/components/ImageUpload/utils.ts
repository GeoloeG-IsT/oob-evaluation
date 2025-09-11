// T076: Image Upload and Gallery - Utility Functions

export const SUPPORTED_FORMATS = [
  'image/jpeg',
  'image/jpg', 
  'image/png',
  'image/webp',
  'image/tiff',
  'image/tif',
  'image/bmp',
  'image/gif'
];

export const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB default

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const formatUploadDate = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diffTime = Math.abs(now.getTime() - date.getTime());
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  
  if (diffDays === 1) {
    return 'Today';
  } else if (diffDays === 2) {
    return 'Yesterday';
  } else if (diffDays <= 7) {
    return `${diffDays - 1} days ago`;
  } else {
    return date.toLocaleDateString();
  }
};

export const validateFile = (file: File): { valid: boolean; error?: string } => {
  // Check file type
  if (!SUPPORTED_FORMATS.includes(file.type)) {
    return {
      valid: false,
      error: `Unsupported file format: ${file.type}. Supported formats: ${SUPPORTED_FORMATS.join(', ')}`
    };
  }
  
  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    return {
      valid: false,
      error: `File too large: ${formatFileSize(file.size)}. Maximum size: ${formatFileSize(MAX_FILE_SIZE)}`
    };
  }
  
  return { valid: true };
};

export const createImagePreview = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

export const generateThumbnail = (
  imageUrl: string,
  maxWidth: number = 200,
  maxHeight: number = 200
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Calculate thumbnail dimensions
      const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
      canvas.width = img.width * ratio;
      canvas.height = img.height * ratio;
      
      // Draw thumbnail
      ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/jpeg', 0.8));
    };
    
    img.onerror = reject;
    img.src = imageUrl;
  });
};

export const uploadImageToServer = async (
  file: File,
  datasetSplit: 'train' | 'val' | 'test' = 'train',
  onProgress?: (progress: number) => void
): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('dataset_split', datasetSplit);
  
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    // Track upload progress
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = (event.loaded / event.total) * 100;
        onProgress(progress);
      }
    };
    
    xhr.onload = () => {
      if (xhr.status === 200) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve(response);
        } catch (error) {
          reject(new Error('Invalid response format'));
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    };
    
    xhr.onerror = () => reject(new Error('Upload failed'));
    
    xhr.open('POST', '/api/v1/images');
    xhr.send(formData);
  });
};

export const fetchImages = async (params?: {
  dataset_split?: 'train' | 'val' | 'test';
  sort_by?: 'upload_date' | 'filename' | 'file_size';
  sort_order?: 'asc' | 'desc';
  offset?: number;
  limit?: number;
}): Promise<any> => {
  const searchParams = new URLSearchParams();
  
  if (params?.dataset_split) {
    searchParams.append('dataset_split', params.dataset_split);
  }
  if (params?.sort_by) {
    searchParams.append('sort_by', params.sort_by);
  }
  if (params?.sort_order) {
    searchParams.append('sort_order', params.sort_order);
  }
  if (params?.offset !== undefined) {
    searchParams.append('offset', params.offset.toString());
  }
  if (params?.limit !== undefined) {
    searchParams.append('limit', params.limit.toString());
  }
  
  const url = `/api/v1/images${searchParams.toString() ? '?' + searchParams.toString() : ''}`;
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch images: ${response.statusText}`);
  }
  
  return response.json();
};

export const deleteImage = async (imageId: string): Promise<void> => {
  const response = await fetch(`/api/v1/images/${imageId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete image: ${response.statusText}`);
  }
};

export const updateImageDatasetSplit = async (
  imageId: string,
  datasetSplit: 'train' | 'val' | 'test'
): Promise<any> => {
  const response = await fetch(`/api/v1/images/${imageId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ dataset_split: datasetSplit }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to update image: ${response.statusText}`);
  }
  
  return response.json();
};

export const calculateSplitDistribution = (images: any[]): {
  train: number;
  val: number;
  test: number;
  total: number;
} => {
  const distribution = images.reduce(
    (acc, image) => {
      acc[image.dataset_split as keyof typeof acc]++;
      acc.total++;
      return acc;
    },
    { train: 0, val: 0, test: 0, total: 0 }
  );
  
  return distribution;
};