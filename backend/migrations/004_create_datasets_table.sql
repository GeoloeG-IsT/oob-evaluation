-- Migration: Create datasets table and dataset_images junction table
-- Description: Creates tables for organizing collections of images by train/validation/test splits

-- Create datasets table
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE CHECK (name != ''),
    description TEXT,
    train_count INTEGER NOT NULL DEFAULT 0 CHECK (train_count >= 0),
    validation_count INTEGER NOT NULL DEFAULT 0 CHECK (validation_count >= 0),
    test_count INTEGER NOT NULL DEFAULT 0 CHECK (test_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT check_at_least_one_split CHECK (
        train_count > 0 OR validation_count > 0 OR test_count > 0
    )
);

-- Create junction table for many-to-many relationship between datasets and images
CREATE TABLE dataset_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL,
    image_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT fk_dataset_images_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    CONSTRAINT fk_dataset_images_image FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    CONSTRAINT unique_dataset_image UNIQUE (dataset_id, image_id)
);

-- Create indexes for performance
CREATE INDEX idx_datasets_name ON datasets (name);
CREATE INDEX idx_datasets_created_at ON datasets (created_at);
CREATE INDEX idx_datasets_updated_at ON datasets (updated_at);
CREATE INDEX idx_datasets_train_count ON datasets (train_count);
CREATE INDEX idx_datasets_validation_count ON datasets (validation_count);
CREATE INDEX idx_datasets_test_count ON datasets (test_count);
CREATE INDEX idx_datasets_metadata ON datasets USING GIN (metadata);

CREATE INDEX idx_dataset_images_dataset_id ON dataset_images (dataset_id);
CREATE INDEX idx_dataset_images_image_id ON dataset_images (image_id);
CREATE INDEX idx_dataset_images_created_at ON dataset_images (created_at);

-- Create trigger to automatically update updated_at timestamp for datasets
CREATE TRIGGER update_datasets_updated_at 
    BEFORE UPDATE ON datasets 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to automatically update dataset split counts
CREATE OR REPLACE FUNCTION update_dataset_counts()
RETURNS TRIGGER AS $$
BEGIN
    -- Update counts for the affected dataset
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        UPDATE datasets SET
            train_count = (
                SELECT COUNT(*) FROM dataset_images di
                JOIN images i ON di.image_id = i.id
                WHERE di.dataset_id = NEW.dataset_id AND i.dataset_split = 'train'
            ),
            validation_count = (
                SELECT COUNT(*) FROM dataset_images di
                JOIN images i ON di.image_id = i.id
                WHERE di.dataset_id = NEW.dataset_id AND i.dataset_split = 'validation'
            ),
            test_count = (
                SELECT COUNT(*) FROM dataset_images di
                JOIN images i ON di.image_id = i.id
                WHERE di.dataset_id = NEW.dataset_id AND i.dataset_split = 'test'
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.dataset_id;
    END IF;
    
    IF TG_OP = 'DELETE' OR TG_OP = 'UPDATE' THEN
        UPDATE datasets SET
            train_count = (
                SELECT COUNT(*) FROM dataset_images di
                JOIN images i ON di.image_id = i.id
                WHERE di.dataset_id = OLD.dataset_id AND i.dataset_split = 'train'
            ),
            validation_count = (
                SELECT COUNT(*) FROM dataset_images di
                JOIN images i ON di.image_id = i.id
                WHERE di.dataset_id = OLD.dataset_id AND i.dataset_split = 'validation'
            ),
            test_count = (
                SELECT COUNT(*) FROM dataset_images di
                JOIN images i ON di.image_id = i.id
                WHERE di.dataset_id = OLD.dataset_id AND i.dataset_split = 'test'
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = OLD.dataset_id;
    END IF;
    
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update dataset counts
CREATE TRIGGER trigger_update_dataset_counts_insert
    AFTER INSERT ON dataset_images
    FOR EACH ROW
    EXECUTE FUNCTION update_dataset_counts();

CREATE TRIGGER trigger_update_dataset_counts_update
    AFTER UPDATE ON dataset_images
    FOR EACH ROW
    EXECUTE FUNCTION update_dataset_counts();

CREATE TRIGGER trigger_update_dataset_counts_delete
    AFTER DELETE ON dataset_images
    FOR EACH ROW
    EXECUTE FUNCTION update_dataset_counts();

-- Add comments for documentation
COMMENT ON TABLE datasets IS 'Collections of images organized by train/validation/test splits';
COMMENT ON COLUMN datasets.id IS 'Unique identifier for the dataset';
COMMENT ON COLUMN datasets.name IS 'Dataset name (must be unique)';
COMMENT ON COLUMN datasets.description IS 'Optional dataset description';
COMMENT ON COLUMN datasets.train_count IS 'Number of training images in the dataset';
COMMENT ON COLUMN datasets.validation_count IS 'Number of validation images in the dataset';
COMMENT ON COLUMN datasets.test_count IS 'Number of test images in the dataset';
COMMENT ON COLUMN datasets.created_at IS 'Timestamp when the dataset was created';
COMMENT ON COLUMN datasets.updated_at IS 'Timestamp when the dataset was last updated';
COMMENT ON COLUMN datasets.metadata IS 'Dataset configuration in JSON format';

COMMENT ON TABLE dataset_images IS 'Junction table for many-to-many relationship between datasets and images';
COMMENT ON COLUMN dataset_images.id IS 'Unique identifier for the dataset-image relationship';
COMMENT ON COLUMN dataset_images.dataset_id IS 'Reference to the dataset';
COMMENT ON COLUMN dataset_images.image_id IS 'Reference to the image';
COMMENT ON COLUMN dataset_images.created_at IS 'Timestamp when the image was added to the dataset';