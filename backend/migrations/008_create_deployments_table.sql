-- Migration: Create deployments table
-- Description: Creates the deployments table for deployed model instances with API endpoints and monitoring

-- Create enum for deployment status
CREATE TYPE deployment_status_enum AS ENUM ('deploying', 'active', 'inactive', 'failed');

-- Create deployments table
CREATE TABLE deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL,
    endpoint_url VARCHAR(1000) NOT NULL CHECK (endpoint_url != ''),
    version VARCHAR(50) NOT NULL CHECK (version != ''),
    status deployment_status_enum NOT NULL DEFAULT 'deploying',
    configuration JSONB NOT NULL CHECK (configuration != '{}'::jsonb),
    performance_monitoring JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Constraints
    CONSTRAINT fk_deployments_model FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE RESTRICT,
    CONSTRAINT unique_endpoint_url UNIQUE (endpoint_url),
    CONSTRAINT unique_model_version UNIQUE (model_id, version),
    CONSTRAINT check_endpoint_url_format CHECK (
        endpoint_url ~* '^https?://.+' OR 
        endpoint_url ~* '^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*(/.*)?$'
    ),
    CONSTRAINT check_version_format CHECK (
        version ~* '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$'
    )
);

-- Create indexes for performance
CREATE INDEX idx_deployments_model_id ON deployments (model_id);
CREATE INDEX idx_deployments_status ON deployments (status);
CREATE INDEX idx_deployments_endpoint_url ON deployments (endpoint_url);
CREATE INDEX idx_deployments_version ON deployments (version);
CREATE INDEX idx_deployments_created_at ON deployments (created_at);
CREATE INDEX idx_deployments_updated_at ON deployments (updated_at);
CREATE INDEX idx_deployments_configuration ON deployments USING GIN (configuration);
CREATE INDEX idx_deployments_performance_monitoring ON deployments USING GIN (performance_monitoring);
CREATE INDEX idx_deployments_metadata ON deployments USING GIN (metadata);

-- Create composite indexes for common queries
CREATE INDEX idx_deployments_model_status ON deployments (model_id, status);
CREATE INDEX idx_deployments_status_created_at ON deployments (status, created_at);

-- Create trigger to automatically update updated_at timestamp
CREATE TRIGGER update_deployments_updated_at 
    BEFORE UPDATE ON deployments 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to validate deployment status transitions
CREATE OR REPLACE FUNCTION validate_deployment_status_transition()
RETURNS TRIGGER AS $$
DECLARE
    old_status deployment_status_enum;
    new_status deployment_status_enum;
BEGIN
    -- Only check transitions on updates
    IF TG_OP = 'UPDATE' THEN
        old_status := OLD.status;
        new_status := NEW.status;
        
        -- Valid status transitions:
        -- deploying -> active, failed
        -- active <-> inactive
        -- failed -> deploying (redeployment)
        -- any -> deploying (redeployment)
        
        IF old_status = new_status THEN
            -- Same status is always allowed
            RETURN NEW;
        END IF;
        
        CASE old_status
            WHEN 'deploying' THEN
                IF new_status NOT IN ('active', 'failed') THEN
                    RAISE EXCEPTION 'Invalid status transition from % to %', old_status, new_status;
                END IF;
            WHEN 'active' THEN
                IF new_status NOT IN ('inactive', 'deploying') THEN
                    RAISE EXCEPTION 'Invalid status transition from % to %', old_status, new_status;
                END IF;
            WHEN 'inactive' THEN
                IF new_status NOT IN ('active', 'deploying') THEN
                    RAISE EXCEPTION 'Invalid status transition from % to %', old_status, new_status;
                END IF;
            WHEN 'failed' THEN
                IF new_status != 'deploying' THEN
                    RAISE EXCEPTION 'Invalid status transition from % to %', old_status, new_status;
                END IF;
        END CASE;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for status transition validation
CREATE TRIGGER trigger_validate_deployment_status_transition
    BEFORE UPDATE ON deployments
    FOR EACH ROW
    EXECUTE FUNCTION validate_deployment_status_transition();

-- Create function to get active deployments for a model
CREATE OR REPLACE FUNCTION get_active_deployments_for_model(p_model_id UUID)
RETURNS TABLE(
    id UUID,
    endpoint_url VARCHAR,
    version VARCHAR,
    configuration JSONB,
    performance_monitoring JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT d.id, d.endpoint_url, d.version, d.configuration, 
           d.performance_monitoring, d.created_at, d.updated_at
    FROM deployments d
    WHERE d.model_id = p_model_id AND d.status = 'active'
    ORDER BY d.created_at DESC;
END;
$$ language 'plpgsql';

-- Create function to get deployment statistics
CREATE OR REPLACE FUNCTION get_deployment_statistics()
RETURNS TABLE(
    total_deployments BIGINT,
    active_deployments BIGINT,
    inactive_deployments BIGINT,
    failed_deployments BIGINT,
    deploying_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_deployments,
        COUNT(*) FILTER (WHERE status = 'active')::BIGINT as active_deployments,
        COUNT(*) FILTER (WHERE status = 'inactive')::BIGINT as inactive_deployments,
        COUNT(*) FILTER (WHERE status = 'failed')::BIGINT as failed_deployments,
        COUNT(*) FILTER (WHERE status = 'deploying')::BIGINT as deploying_count
    FROM deployments;
END;
$$ language 'plpgsql';

-- Create function to update performance monitoring data
CREATE OR REPLACE FUNCTION update_deployment_performance(
    p_deployment_id UUID,
    p_performance_data JSONB
)
RETURNS BOOLEAN AS $$
DECLARE
    deployment_exists BOOLEAN;
BEGIN
    -- Check if deployment exists and is active
    SELECT EXISTS(
        SELECT 1 FROM deployments 
        WHERE id = p_deployment_id AND status = 'active'
    ) INTO deployment_exists;
    
    IF NOT deployment_exists THEN
        RETURN FALSE;
    END IF;
    
    -- Update performance monitoring data
    UPDATE deployments 
    SET 
        performance_monitoring = performance_monitoring || p_performance_data,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = p_deployment_id;
    
    RETURN TRUE;
END;
$$ language 'plpgsql';

-- Create view for deployment monitoring dashboard
CREATE VIEW deployment_monitoring_view AS
SELECT 
    d.id,
    d.model_id,
    m.name as model_name,
    m.type as model_type,
    m.framework as model_framework,
    d.endpoint_url,
    d.version,
    d.status,
    d.performance_monitoring,
    d.created_at,
    d.updated_at,
    -- Extract key performance metrics if they exist
    (d.performance_monitoring->>'requests_per_minute')::FLOAT as requests_per_minute,
    (d.performance_monitoring->>'average_response_time_ms')::FLOAT as avg_response_time_ms,
    (d.performance_monitoring->>'error_rate_percent')::FLOAT as error_rate_percent,
    (d.performance_monitoring->>'cpu_usage_percent')::FLOAT as cpu_usage_percent,
    (d.performance_monitoring->>'memory_usage_mb')::FLOAT as memory_usage_mb
FROM deployments d
JOIN models m ON d.model_id = m.id
WHERE d.status IN ('active', 'inactive');

-- Add comments for documentation
COMMENT ON TABLE deployments IS 'Deployed model instances with API endpoints and monitoring';
COMMENT ON COLUMN deployments.id IS 'Unique identifier for the deployment';
COMMENT ON COLUMN deployments.model_id IS 'Reference to the deployed model';
COMMENT ON COLUMN deployments.endpoint_url IS 'API endpoint URL for the deployment';
COMMENT ON COLUMN deployments.version IS 'Deployment version (semantic versioning)';
COMMENT ON COLUMN deployments.status IS 'Current deployment status';
COMMENT ON COLUMN deployments.configuration IS 'Deployment settings in JSON format';
COMMENT ON COLUMN deployments.performance_monitoring IS 'Usage and performance metrics in JSON format';
COMMENT ON COLUMN deployments.created_at IS 'Timestamp when the deployment was created';
COMMENT ON COLUMN deployments.updated_at IS 'Timestamp when the deployment was last updated';
COMMENT ON COLUMN deployments.metadata IS 'Additional deployment data in JSON format';

COMMENT ON VIEW deployment_monitoring_view IS 'Monitoring dashboard view with key deployment metrics and model information';