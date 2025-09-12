-- Migration: Create annotations table
-- Description: Creates the annotations table for storing object/segment annotations with coordinates and labels

-- Create enum for annotation creation method
CREATE TYPE creation_method_enum AS ENUM ('user', 'model');

-- Create annotations table
CREATE TABLE annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL,
    bounding_boxes JSONB DEFAULT '[]'::jsonb,
    segments JSONB DEFAULT '[]'::jsonb,
    class_labels TEXT[] NOT NULL CHECK (array_length(class_labels, 1) > 0),
    confidence_scores FLOAT[] DEFAULT '{}',
    creation_method creation_method_enum NOT NULL DEFAULT 'user',
    model_id UUID,
    user_tag VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT fk_annotations_image FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    CONSTRAINT fk_annotations_model FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL,
    CONSTRAINT check_annotation_data CHECK (
        (bounding_boxes != '[]'::jsonb) OR (segments != '[]'::jsonb)
    ),
    CONSTRAINT check_confidence_scores_length CHECK (
        confidence_scores IS NULL OR 
        array_length(confidence_scores, 1) = array_length(class_labels, 1)
    ),
    CONSTRAINT check_model_creation_method CHECK (
        (creation_method = 'user' AND model_id IS NULL) OR
        (creation_method = 'model' AND model_id IS NOT NULL)
    )
);

-- Create indexes for performance
CREATE INDEX idx_annotations_image_id ON annotations (image_id);
CREATE INDEX idx_annotations_model_id ON annotations (model_id);
CREATE INDEX idx_annotations_creation_method ON annotations (creation_method);
CREATE INDEX idx_annotations_created_at ON annotations (created_at);
CREATE INDEX idx_annotations_user_tag ON annotations (user_tag);
CREATE INDEX idx_annotations_class_labels ON annotations USING GIN (class_labels);
CREATE INDEX idx_annotations_bounding_boxes ON annotations USING GIN (bounding_boxes);
CREATE INDEX idx_annotations_segments ON annotations USING GIN (segments);
CREATE INDEX idx_annotations_metadata ON annotations USING GIN (metadata);

-- Add comments for documentation
COMMENT ON TABLE annotations IS 'Stores object/segment annotations with coordinates, labels, and creation metadata';
COMMENT ON COLUMN annotations.id IS 'Unique identifier for the annotation';
COMMENT ON COLUMN annotations.image_id IS 'Reference to the associated image';
COMMENT ON COLUMN annotations.bounding_boxes IS 'Object detection bounding boxes in JSON format';
COMMENT ON COLUMN annotations.segments IS 'Segmentation masks/polygons in JSON format';
COMMENT ON COLUMN annotations.class_labels IS 'Array of object class names';
COMMENT ON COLUMN annotations.confidence_scores IS 'Array of prediction confidence scores (if model-generated)';
COMMENT ON COLUMN annotations.creation_method IS 'How the annotation was created (user or model)';
COMMENT ON COLUMN annotations.model_id IS 'Reference to the model that created this annotation (if applicable)';
COMMENT ON COLUMN annotations.user_tag IS 'User identifier for manual annotations';
COMMENT ON COLUMN annotations.created_at IS 'Timestamp when the annotation was created';
COMMENT ON COLUMN annotations.metadata IS 'Additional annotation data in JSON format';