-- Migration: Create training_jobs table
-- Description: Creates the training_jobs table for model training/fine-tuning tasks

-- Create enum for training job status
CREATE TYPE training_job_status_enum AS ENUM ('queued', 'running', 'completed', 'failed', 'cancelled');

-- Create training_jobs table
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    base_model_id UUID NOT NULL,
    dataset_id UUID NOT NULL,
    status training_job_status_enum NOT NULL DEFAULT 'queued',
    progress_percentage FLOAT NOT NULL DEFAULT 0.0 CHECK (progress_percentage >= 0.0 AND progress_percentage <= 100.0),
    hyperparameters JSONB NOT NULL CHECK (hyperparameters != '{}'::jsonb),
    execution_logs TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    result_model_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT fk_training_jobs_base_model FOREIGN KEY (base_model_id) REFERENCES models(id) ON DELETE RESTRICT,
    CONSTRAINT fk_training_jobs_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE RESTRICT,
    CONSTRAINT fk_training_jobs_result_model FOREIGN KEY (result_model_id) REFERENCES models(id) ON DELETE SET NULL,
    CONSTRAINT check_completed_has_result CHECK (
        (status != 'completed') OR (status = 'completed' AND result_model_id IS NOT NULL)
    ),
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

-- Create indexes for performance
CREATE INDEX idx_training_jobs_base_model_id ON training_jobs (base_model_id);
CREATE INDEX idx_training_jobs_dataset_id ON training_jobs (dataset_id);
CREATE INDEX idx_training_jobs_result_model_id ON training_jobs (result_model_id);
CREATE INDEX idx_training_jobs_status ON training_jobs (status);
CREATE INDEX idx_training_jobs_created_at ON training_jobs (created_at);
CREATE INDEX idx_training_jobs_start_time ON training_jobs (start_time);
CREATE INDEX idx_training_jobs_end_time ON training_jobs (end_time);
CREATE INDEX idx_training_jobs_progress ON training_jobs (progress_percentage);
CREATE INDEX idx_training_jobs_hyperparameters ON training_jobs USING GIN (hyperparameters);
CREATE INDEX idx_training_jobs_metadata ON training_jobs USING GIN (metadata);

-- Create composite indexes for common queries
CREATE INDEX idx_training_jobs_status_created_at ON training_jobs (status, created_at);
CREATE INDEX idx_training_jobs_base_model_status ON training_jobs (base_model_id, status);
CREATE INDEX idx_training_jobs_dataset_status ON training_jobs (dataset_id, status);

-- Create function to validate training job status transitions
CREATE OR REPLACE FUNCTION validate_training_job_status_transition()
RETURNS TRIGGER AS $$
DECLARE
    old_status training_job_status_enum;
    new_status training_job_status_enum;
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
CREATE TRIGGER trigger_validate_training_job_status_transition
    BEFORE UPDATE ON training_jobs
    FOR EACH ROW
    EXECUTE FUNCTION validate_training_job_status_transition();

-- Add comments for documentation
COMMENT ON TABLE training_jobs IS 'Model training/fine-tuning tasks with status and configuration';
COMMENT ON COLUMN training_jobs.id IS 'Unique identifier for the training job';
COMMENT ON COLUMN training_jobs.base_model_id IS 'Reference to the base model for training';
COMMENT ON COLUMN training_jobs.dataset_id IS 'Reference to the training dataset';
COMMENT ON COLUMN training_jobs.status IS 'Current status of the training job';
COMMENT ON COLUMN training_jobs.progress_percentage IS 'Training progress (0.0 to 100.0)';
COMMENT ON COLUMN training_jobs.hyperparameters IS 'Training configuration in JSON format';
COMMENT ON COLUMN training_jobs.execution_logs IS 'Training logs and output';
COMMENT ON COLUMN training_jobs.start_time IS 'Timestamp when training started';
COMMENT ON COLUMN training_jobs.end_time IS 'Timestamp when training completed/failed';
COMMENT ON COLUMN training_jobs.result_model_id IS 'Reference to the resulting trained model';
COMMENT ON COLUMN training_jobs.created_at IS 'Timestamp when the job was created';
COMMENT ON COLUMN training_jobs.metadata IS 'Additional training metadata in JSON format';