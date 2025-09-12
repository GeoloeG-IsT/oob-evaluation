-- Migration: Create images table
-- Description: Creates the images table for storing uploaded image files with metadata

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum for dataset split
CREATE TYPE dataset_split_enum AS ENUM ('train', 'validation', 'test');

-- Create images table
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL CHECK (filename != ''),
    file_path VARCHAR(1000) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size > 0),
    format VARCHAR(20) NOT NULL CHECK (format != ''),
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    dataset_split dataset_split_enum NOT NULL DEFAULT 'train',
    upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT unique_filename_path UNIQUE (filename, file_path)
);

-- Create indexes for performance
CREATE INDEX idx_images_filename ON images (filename);
CREATE INDEX idx_images_dataset_split ON images (dataset_split);
CREATE INDEX idx_images_upload_timestamp ON images (upload_timestamp);
CREATE INDEX idx_images_format ON images (format);
CREATE INDEX idx_images_metadata ON images USING GIN (metadata);

-- Add comments for documentation
COMMENT ON TABLE images IS 'Stores uploaded image files with associated metadata and dataset organization';
COMMENT ON COLUMN images.id IS 'Unique identifier for the image';
COMMENT ON COLUMN images.filename IS 'Original filename of the uploaded image';
COMMENT ON COLUMN images.file_path IS 'Storage path or URL where the image is stored';
COMMENT ON COLUMN images.file_size IS 'File size in bytes';
COMMENT ON COLUMN images.format IS 'Image format (JPEG, PNG, TIFF, etc.)';
COMMENT ON COLUMN images.width IS 'Image width in pixels';
COMMENT ON COLUMN images.height IS 'Image height in pixels';
COMMENT ON COLUMN images.dataset_split IS 'Dataset split assignment (train/validation/test)';
COMMENT ON COLUMN images.upload_timestamp IS 'Timestamp when the image was uploaded';
COMMENT ON COLUMN images.metadata IS 'Additional image metadata in JSON format';