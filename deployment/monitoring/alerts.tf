# Cloud Monitoring alert policies for ML Evaluation Platform

# High error rate alert
resource "google_monitoring_alert_policy" "high_error_rate" {
  project      = var.project_id
  display_name = "ML Eval - High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "High error rate condition"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND resource.labels.service_name=~\"ml-eval-.*\" AND metric.labels.response_code_class!=\"2xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.1  # 10% error rate

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }

      trigger {
        count = 1
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"  # 30 minutes
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Error rate is above 10% for ML Evaluation Platform services. Check service logs and health endpoints."
    mime_type = "text/markdown"
  }
}

# High response time alert
resource "google_monitoring_alert_policy" "high_response_time" {
  project      = var.project_id
  display_name = "ML Eval - High Response Time"
  combiner     = "OR"

  conditions {
    display_name = "High response time condition"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\" AND resource.labels.service_name=~\"ml-eval-.*\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 5000  # 5 seconds

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
        group_by_fields      = ["resource.label.service_name"]
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "95th percentile response time is above 5 seconds. This may indicate performance issues or resource constraints."
    mime_type = "text/markdown"
  }
}

# High memory utilization alert
resource "google_monitoring_alert_policy" "high_memory_usage" {
  project      = var.project_id
  display_name = "ML Eval - High Memory Usage"
  combiner     = "OR"

  conditions {
    display_name = "High memory usage condition"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\" AND resource.labels.service_name=~\"ml-eval-.*\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.85  # 85%

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }

      trigger {
        count = 2
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Memory utilization is above 85%. Consider increasing memory allocation or optimizing memory usage."
    mime_type = "text/markdown"
  }
}

# Database connection alert
resource "google_monitoring_alert_policy" "database_connections" {
  project      = var.project_id
  display_name = "ML Eval - High Database Connections"
  combiner     = "OR"

  conditions {
    display_name = "High database connections condition"
    condition_threshold {
      filter          = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/postgresql/num_backends\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 180  # 90% of max connections (200)

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Database connection count is approaching the limit. Check for connection leaks or consider connection pooling optimization."
    mime_type = "text/markdown"
  }
}

# Database disk usage alert
resource "google_monitoring_alert_policy" "database_disk_usage" {
  project      = var.project_id
  display_name = "ML Eval - Database Disk Usage"
  combiner     = "OR"

  conditions {
    display_name = "High database disk usage condition"
    condition_threshold {
      filter          = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/disk/utilization\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8  # 80%

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Database disk utilization is above 80%. Consider cleanup, archiving, or increasing disk size."
    mime_type = "text/markdown"
  }
}

# Redis memory usage alert
resource "google_monitoring_alert_policy" "redis_memory_usage" {
  project      = var.project_id
  display_name = "ML Eval - Redis Memory Usage"
  combiner     = "OR"

  conditions {
    display_name = "High Redis memory usage condition"
    condition_threshold {
      filter          = "resource.type=\"redis_instance\" AND metric.type=\"redis.googleapis.com/stats/memory/usage_ratio\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.85  # 85%

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Redis memory usage is above 85%. Check for memory leaks or consider increasing Redis instance size."
    mime_type = "text/markdown"
  }
}

# Service down alert
resource "google_monitoring_alert_policy" "service_down" {
  project      = var.project_id
  display_name = "ML Eval - Service Down"
  combiner     = "OR"

  conditions {
    display_name = "Service availability condition"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND resource.labels.service_name=~\"ml-eval-(backend|frontend)\""
      duration        = "300s"
      comparison      = "COMPARISON_LESS_THAN"
      threshold_value = 0.1  # Very low request rate indicates service might be down

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["resource.label.service_name"]
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Critical service (backend or frontend) appears to be down or receiving very low traffic. Immediate investigation required."
    mime_type = "text/markdown"
  }
}

# ML training job failure alert
resource "google_monitoring_alert_policy" "ml_training_failure" {
  project      = var.project_id
  display_name = "ML Eval - Training Job Failures"
  combiner     = "OR"

  conditions {
    display_name = "Training job failure condition"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND resource.labels.service_name=\"ml-eval-celery-training\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "60s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 3  # 3 failures in 1 minute

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Multiple ML training job failures detected. Check training worker logs and resource availability."
    mime_type = "text/markdown"
  }
}

# Budget alert (cost management)
resource "google_monitoring_alert_policy" "budget_alert" {
  project      = var.project_id
  display_name = "ML Eval - Budget Alert"
  combiner     = "OR"

  conditions {
    display_name = "High spending condition"
    condition_threshold {
      filter          = "metric.type=\"billing.googleapis.com/billing/total_cost\" AND resource.type=\"billing_account\""
      duration        = "3600s"  # 1 hour
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = var.daily_budget_threshold

      aggregations {
        alignment_period   = "3600s"
        per_series_aligner = "ALIGN_DELTA"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "Daily spending threshold exceeded. Review resource usage and costs to ensure budget compliance."
    mime_type = "text/markdown"
  }
}

# Custom log-based alert for application errors
resource "google_logging_metric" "application_errors" {
  name   = "ml_eval_application_errors"
  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=~\"ml-eval-.*\" AND (severity=\"ERROR\" OR jsonPayload.level=\"error\" OR textPayload=~\"ERROR\")"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    display_name = "ML Eval Application Errors"
  }

  label_extractors = {
    "service_name" = "EXTRACT(resource.labels.service_name)"
    "error_type"   = "EXTRACT(jsonPayload.error_type)"
  }
}

resource "google_monitoring_alert_policy" "application_errors" {
  project      = var.project_id
  display_name = "ML Eval - Application Errors"
  combiner     = "OR"

  conditions {
    display_name = "Application error rate condition"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/ml_eval_application_errors\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 10  # 10 errors in 5 minutes

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }

      trigger {
        count = 1
      }
    }
  }

  notification_channels = var.notification_channels

  documentation {
    content = "High rate of application errors detected in logs. Check application logs for specific error details."
    mime_type = "text/markdown"
  }
}