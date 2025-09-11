-- Migration: Create inference_jobs table
-- Description: Creates the inference_jobs table for batch inference tasks on multiple images

-- Create enum for inference job status
CREATE TYPE inference_job_status_enum AS ENUM ('queued', 'running', 'completed', 'failed', 'cancelled');

-- Create inference_jobs table
CREATE TABLE inference_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL,
    target_images UUID[] NOT NULL CHECK (array_length(target_images, 1) > 0),
    status inference_job_status_enum NOT NULL DEFAULT 'queued',
    progress_percentage FLOAT NOT NULL DEFAULT 0.0 CHECK (progress_percentage >= 0.0 AND progress_percentage <= 100.0),
    results JSONB DEFAULT '{}'::jsonb,
    execution_logs TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT fk_inference_jobs_model FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE RESTRICT,
    CONSTRAINT check_time_order CHECK (
        start_time IS NULL OR end_time IS NULL OR start_time <= end_time
    ),
    CONSTRAINT check_running_has_start_time CHECK (
        (status NOT IN ('running', 'completed', 'failed', 'cancelled')) OR start_time IS NOT NULL
    ),
    CONSTRAINT check_finished_has_end_time CHECK (
        (status NOT IN ('completed', 'failed', 'cancelled')) OR end_time IS NOT NULL
    )
);

-- Create function to validate target image references
CREATE OR REPLACE FUNCTION validate_inference_job_target_images()
RETURNS TRIGGER AS $$
DECLARE
    image_id UUID;
    invalid_count INTEGER;
BEGIN
    -- Check that all target images exist
    SELECT COUNT(*) INTO invalid_count
    FROM unnest(NEW.target_images) AS image_id
    WHERE image_id NOT IN (SELECT id FROM images);
    
    IF invalid_count > 0 THEN
        RAISE EXCEPTION 'Invalid image IDs found in target_images array';
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to validate target images
CREATE TRIGGER trigger_validate_inference_job_target_images
    BEFORE INSERT OR UPDATE ON inference_jobs
    FOR EACH ROW
    EXECUTE FUNCTION validate_inference_job_target_images();

-- Create indexes for performance
CREATE INDEX idx_inference_jobs_model_id ON inference_jobs (model_id);
CREATE INDEX idx_inference_jobs_status ON inference_jobs (status);
CREATE INDEX idx_inference_jobs_created_at ON inference_jobs (created_at);
CREATE INDEX idx_inference_jobs_start_time ON inference_jobs (start_time);
CREATE INDEX idx_inference_jobs_end_time ON inference_jobs (end_time);
CREATE INDEX idx_inference_jobs_progress ON inference_jobs (progress_percentage);
CREATE INDEX idx_inference_jobs_target_images ON inference_jobs USING GIN (target_images);
CREATE INDEX idx_inference_jobs_results ON inference_jobs USING GIN (results);
CREATE INDEX idx_inference_jobs_metadata ON inference_jobs USING GIN (metadata);

-- Create composite indexes for common queries
CREATE INDEX idx_inference_jobs_status_created_at ON inference_jobs (status, created_at);
CREATE INDEX idx_inference_jobs_model_status ON inference_jobs (model_id, status);

-- Create function to validate inference job status transitions
CREATE OR REPLACE FUNCTION validate_inference_job_status_transition()
RETURNS TRIGGER AS $$
DECLARE
    old_status inference_job_status_enum;
    new_status inference_job_status_enum;
BEGIN
    -- Only check transitions on updates
    IF TG_OP = 'UPDATE' THEN
        old_status := OLD.status;
        new_status := NEW.status;
        
        -- Valid status transitions:
        -- queued -> running, cancelled
        -- running -> completed, failed, cancelled  
        -- completed, failed, cancelled -> (no transitions allowed)
        
        IF old_status = new_status THEN
            -- Same status is always allowed
            RETURN NEW;
        END IF;
        
        CASE old_status
            WHEN 'queued' THEN
                IF new_status NOT IN ('running', 'cancelled') THEN
                    RAISE EXCEPTION 'Invalid status transition from % to %', old_status, new_status;
                END IF;
            WHEN 'running' THEN
                IF new_status NOT IN ('completed', 'failed', 'cancelled') THEN
                    RAISE EXCEPTION 'Invalid status transition from % to %', old_status, new_status;
                END IF;
            WHEN 'completed', 'failed', 'cancelled' THEN
                RAISE EXCEPTION 'Cannot transition from terminal status %', old_status;
        END CASE;
        
        -- Set timestamps based on status transitions
        IF old_status = 'queued' AND new_status = 'running' THEN
            NEW.start_time = CURRENT_TIMESTAMP;
        ELSIF new_status IN ('completed', 'failed', 'cancelled') AND OLD.end_time IS NULL THEN
            NEW.end_time = CURRENT_TIMESTAMP;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for status transition validation
CREATE TRIGGER trigger_validate_inference_job_status_transition
    BEFORE UPDATE ON inference_jobs
    FOR EACH ROW
    EXECUTE FUNCTION validate_inference_job_status_transition();

-- Create function to automatically create annotations from inference results
CREATE OR REPLACE FUNCTION create_annotations_from_inference()
RETURNS TRIGGER AS $$
DECLARE
    image_id UUID;
    image_results JSONB;
BEGIN
    -- Only process when job completes successfully
    IF NEW.status = 'completed' AND OLD.status != 'completed' AND NEW.results IS NOT NULL THEN
        -- Loop through each image in the results
        FOR image_id, image_results IN SELECT * FROM jsonb_each(NEW.results) LOOP
            -- Insert annotation for this image if results exist
            IF image_results IS NOT NULL AND image_results != 'null'::jsonb THEN
                INSERT INTO annotations (
                    image_id,
                    bounding_boxes,
                    segments,
                    class_labels,
                    confidence_scores,
                    creation_method,
                    model_id,
                    created_at,
                    metadata
                ) VALUES (
                    image_id::UUID,
                    COALESCE(image_results->>'bounding_boxes', '[]')::jsonb,
                    COALESCE(image_results->>'segments', '[]')::jsonb,
                    COALESCE(
                        (SELECT array_agg(value::text) FROM jsonb_array_elements_text(image_results->'class_labels')),
                        '{}'::text[]
                    ),
                    COALESCE(
                        (SELECT array_agg(value::float) FROM jsonb_array_elements_text(image_results->'confidence_scores')),
                        '{}'::float[]
                    ),
                    'model',
                    NEW.model_id,
                    CURRENT_TIMESTAMP,
                    COALESCE(image_results->'metadata', '{}'::jsonb)
                );
            END IF;
        END LOOP;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically create annotations
CREATE TRIGGER trigger_create_annotations_from_inference
    AFTER UPDATE ON inference_jobs
    FOR EACH ROW
    EXECUTE FUNCTION create_annotations_from_inference();

-- Add comments for documentation
COMMENT ON TABLE inference_jobs IS 'Batch inference tasks on multiple images';
COMMENT ON COLUMN inference_jobs.id IS 'Unique identifier for the inference job';
COMMENT ON COLUMN inference_jobs.model_id IS 'Reference to the model used for inference';
COMMENT ON COLUMN inference_jobs.target_images IS 'Array of image IDs to process';
COMMENT ON COLUMN inference_jobs.status IS 'Current status of the inference job';
COMMENT ON COLUMN inference_jobs.progress_percentage IS 'Processing progress (0.0 to 100.0)';
COMMENT ON COLUMN inference_jobs.results IS 'Inference results in JSON format';
COMMENT ON COLUMN inference_jobs.execution_logs IS 'Processing logs and output';
COMMENT ON COLUMN inference_jobs.start_time IS 'Timestamp when processing started';
COMMENT ON COLUMN inference_jobs.end_time IS 'Timestamp when processing completed/failed';
COMMENT ON COLUMN inference_jobs.created_at IS 'Timestamp when the job was created';
COMMENT ON COLUMN inference_jobs.metadata IS 'Additional inference metadata in JSON format';