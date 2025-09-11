-- Migration: Create models table
-- Description: Creates the models table for storing AI model metadata and status

-- Create enums for model fields
CREATE TYPE model_type_enum AS ENUM ('detection', 'segmentation');
CREATE TYPE model_framework_enum AS ENUM ('YOLO11', 'YOLO12', 'RT-DETR', 'SAM2');
CREATE TYPE training_status_enum AS ENUM ('pre-trained', 'training', 'trained', 'failed');

-- Create models table
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL CHECK (name != ''),
    type model_type_enum NOT NULL,
    variant VARCHAR(50) NOT NULL CHECK (variant != ''),
    version VARCHAR(50) NOT NULL CHECK (version != ''),
    framework model_framework_enum NOT NULL,
    model_path VARCHAR(1000) NOT NULL CHECK (model_path != ''),
    training_status training_status_enum NOT NULL DEFAULT 'pre-trained',
    performance_metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT unique_model_name_version UNIQUE (name, version),
    CONSTRAINT check_variant_values CHECK (
        variant IN ('nano', 'small', 'medium', 'large', 'xl', 'extra-large', 
                   'R18', 'R34', 'R50', 'R101', 'RF-DETR-Nano', 'RF-DETR-Small', 'RF-DETR-Medium',
                   'Tiny', 'Small', 'Base+', 'Large')
    )
);

-- Create indexes for performance
CREATE INDEX idx_models_name ON models (name);
CREATE INDEX idx_models_type ON models (type);
CREATE INDEX idx_models_framework ON models (framework);
CREATE INDEX idx_models_training_status ON models (training_status);
CREATE INDEX idx_models_variant ON models (variant);
CREATE INDEX idx_models_created_at ON models (created_at);
CREATE INDEX idx_models_updated_at ON models (updated_at);
CREATE INDEX idx_models_performance_metrics ON models USING GIN (performance_metrics);
CREATE INDEX idx_models_metadata ON models USING GIN (metadata);

-- Create trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_models_updated_at 
    BEFORE UPDATE ON models 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE models IS 'Stores AI model metadata, performance metrics, and training status';
COMMENT ON COLUMN models.id IS 'Unique identifier for the model';
COMMENT ON COLUMN models.name IS 'Model name';
COMMENT ON COLUMN models.type IS 'Model type (detection or segmentation)';
COMMENT ON COLUMN models.variant IS 'Model size variant (nano, small, medium, large, xl, etc.)';
COMMENT ON COLUMN models.version IS 'Model version';
COMMENT ON COLUMN models.framework IS 'Model framework (YOLO11, YOLO12, RT-DETR, SAM2)';
COMMENT ON COLUMN models.model_path IS 'Storage path for model files';
COMMENT ON COLUMN models.training_status IS 'Current training status';
COMMENT ON COLUMN models.performance_metrics IS 'mAP, accuracy, speed metrics in JSON format';
COMMENT ON COLUMN models.created_at IS 'Timestamp when the model was created';
COMMENT ON COLUMN models.updated_at IS 'Timestamp when the model was last updated';
COMMENT ON COLUMN models.metadata IS 'Framework-specific configuration in JSON format';