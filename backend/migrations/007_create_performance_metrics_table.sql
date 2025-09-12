-- Migration: Create performance_metrics table
-- Description: Creates the performance_metrics table for evaluation results with accuracy scores and execution times

-- Create enum for metric types
CREATE TYPE metric_type_enum AS ENUM ('mAP', 'IoU', 'precision', 'recall', 'F1', 'execution_time');

-- Create performance_metrics table (regular table, partitioning can be added later if needed)
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL,
    dataset_id UUID,
    metric_type metric_type_enum NOT NULL,
    metric_value FLOAT NOT NULL CHECK (metric_value >= 0.0),
    threshold FLOAT CHECK (threshold >= 0.0 AND threshold <= 1.0),
    class_name VARCHAR(255),
    evaluation_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT fk_performance_metrics_model FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
    CONSTRAINT fk_performance_metrics_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    CONSTRAINT check_map_threshold CHECK (
        metric_type != 'mAP' OR threshold IS NOT NULL
    ),
    CONSTRAINT check_execution_time_no_threshold CHECK (
        metric_type != 'execution_time' OR threshold IS NULL
    )
);

-- Create indexes for performance
CREATE INDEX idx_performance_metrics_model_id ON performance_metrics (model_id);
CREATE INDEX idx_performance_metrics_dataset_id ON performance_metrics (dataset_id);
CREATE INDEX idx_performance_metrics_metric_type ON performance_metrics (metric_type);
CREATE INDEX idx_performance_metrics_evaluation_timestamp ON performance_metrics (evaluation_timestamp);
CREATE INDEX idx_performance_metrics_class_name ON performance_metrics (class_name);
CREATE INDEX idx_performance_metrics_metric_value ON performance_metrics (metric_value);
CREATE INDEX idx_performance_metrics_threshold ON performance_metrics (threshold);
CREATE INDEX idx_performance_metrics_metadata ON performance_metrics USING GIN (metadata);

-- Create composite indexes for common queries
CREATE INDEX idx_performance_metrics_model_metric_type ON performance_metrics (model_id, metric_type);
CREATE INDEX idx_performance_metrics_model_dataset ON performance_metrics (model_id, dataset_id);
CREATE INDEX idx_performance_metrics_model_class ON performance_metrics (model_id, class_name);
CREATE INDEX idx_performance_metrics_metric_timestamp ON performance_metrics (metric_type, evaluation_timestamp);


-- Create function to get latest metric for a model
CREATE OR REPLACE FUNCTION get_latest_performance_metric(
    p_model_id UUID,
    p_metric_type metric_type_enum,
    p_dataset_id UUID DEFAULT NULL,
    p_class_name VARCHAR DEFAULT NULL
)
RETURNS TABLE(
    id UUID,
    metric_value FLOAT,
    threshold FLOAT,
    evaluation_timestamp TIMESTAMPTZ,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT pm.id, pm.metric_value, pm.threshold, pm.evaluation_timestamp, pm.metadata
    FROM performance_metrics pm
    WHERE pm.model_id = p_model_id
      AND pm.metric_type = p_metric_type
      AND (p_dataset_id IS NULL OR pm.dataset_id = p_dataset_id)
      AND (p_class_name IS NULL OR pm.class_name = p_class_name)
    ORDER BY pm.evaluation_timestamp DESC
    LIMIT 1;
END;
$$ language 'plpgsql';

-- Create function to get metric trends over time
CREATE OR REPLACE FUNCTION get_performance_metric_trends(
    p_model_id UUID,
    p_metric_type metric_type_enum,
    p_days INTEGER DEFAULT 30,
    p_dataset_id UUID DEFAULT NULL,
    p_class_name VARCHAR DEFAULT NULL
)
RETURNS TABLE(
    evaluation_date DATE,
    avg_metric_value FLOAT,
    min_metric_value FLOAT,
    max_metric_value FLOAT,
    metric_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pm.evaluation_timestamp::DATE as evaluation_date,
        AVG(pm.metric_value)::FLOAT as avg_metric_value,
        MIN(pm.metric_value)::FLOAT as min_metric_value,
        MAX(pm.metric_value)::FLOAT as max_metric_value,
        COUNT(*)::BIGINT as metric_count
    FROM performance_metrics pm
    WHERE pm.model_id = p_model_id
      AND pm.metric_type = p_metric_type
      AND pm.evaluation_timestamp >= CURRENT_DATE - INTERVAL '%s days' % p_days
      AND (p_dataset_id IS NULL OR pm.dataset_id = p_dataset_id)
      AND (p_class_name IS NULL OR pm.class_name = p_class_name)
    GROUP BY pm.evaluation_timestamp::DATE
    ORDER BY evaluation_date;
END;
$$ language 'plpgsql';

-- Add comments for documentation
COMMENT ON TABLE performance_metrics IS 'Evaluation results with accuracy scores and execution times';
COMMENT ON COLUMN performance_metrics.id IS 'Unique identifier for the performance metric';
COMMENT ON COLUMN performance_metrics.model_id IS 'Reference to the evaluated model';
COMMENT ON COLUMN performance_metrics.dataset_id IS 'Reference to the test dataset (optional)';
COMMENT ON COLUMN performance_metrics.metric_type IS 'Type of metric (mAP, IoU, precision, recall, F1, execution_time)';
COMMENT ON COLUMN performance_metrics.metric_value IS 'The measured metric value';
COMMENT ON COLUMN performance_metrics.threshold IS 'IoU threshold for mAP calculations (optional)';
COMMENT ON COLUMN performance_metrics.class_name IS 'Class name for class-specific metrics (optional)';
COMMENT ON COLUMN performance_metrics.evaluation_timestamp IS 'Timestamp when the metric was calculated';
COMMENT ON COLUMN performance_metrics.metadata IS 'Additional metric details in JSON format';