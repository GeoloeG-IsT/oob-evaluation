# Cloud Monitoring dashboards for ML Evaluation Platform

# Main application dashboard
resource "google_monitoring_dashboard" "ml_eval_main" {
  project      = var.project_id
  display_name = "ML Evaluation Platform - Main Dashboard"

  dashboard_json = jsonencode({
    displayName = "ML Evaluation Platform - Main Dashboard"
    mosaicLayout = {
      tiles = [
        # Cloud Run Services Health
        {
          width  = 6
          height = 4
          widget = {
            title = "Cloud Run Services - Request Rate"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=~\"ml-eval-.*\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_RATE"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields = ["resource.label.service_name"]
                      }
                    }
                    unitOverride = "1/s"
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Requests per second"
                scale = "LINEAR"
              }
              chartOptions = {
                mode = "COLOR"
              }
            }
          }
        },
        # Error Rate
        {
          xPos   = 6
          width  = 6
          height = 4
          widget = {
            title = "Error Rate by Service"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND resource.labels.service_name=~\"ml-eval-.*\" AND metric.labels.response_code_class!=\"2xx\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_RATE"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields = ["resource.label.service_name"]
                      }
                    }
                    unitOverride = "1"
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Error rate"
                scale = "LINEAR"
              }
            }
          }
        },
        # Response Time
        {
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Response Time (95th percentile)"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\" AND resource.labels.service_name=~\"ml-eval-.*\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_DELTA"
                        crossSeriesReducer = "REDUCE_PERCENTILE_95"
                        groupByFields = ["resource.label.service_name"]
                      }
                    }
                    unitOverride = "ms"
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Latency (ms)"
                scale = "LINEAR"
              }
            }
          }
        },
        # Memory Usage
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Memory Utilization"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\" AND resource.labels.service_name=~\"ml-eval-.*\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_MEAN"
                        groupByFields = ["resource.label.service_name"]
                      }
                    }
                    unitOverride = "1"
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Memory utilization (%)"
                scale = "LINEAR"
              }
            }
          }
        },
        # CPU Usage
        {
          yPos   = 8
          width  = 6
          height = 4
          widget = {
            title = "CPU Utilization"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\" AND resource.labels.service_name=~\"ml-eval-.*\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_MEAN"
                        groupByFields = ["resource.label.service_name"]
                      }
                    }
                    unitOverride = "1"
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "CPU utilization (%)"
                scale = "LINEAR"
              }
            }
          }
        },
        # Instance Count
        {
          xPos   = 6
          yPos   = 8
          width  = 6
          height = 4
          widget = {
            title = "Active Instances"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/container/instance_count\" AND resource.labels.service_name=~\"ml-eval-.*\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields = ["resource.label.service_name"]
                      }
                    }
                    unitOverride = "1"
                  }
                  plotType = "STACKED_AREA"
                  targetAxis = "Y1"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Instance count"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}

# Database dashboard
resource "google_monitoring_dashboard" "ml_eval_database" {
  project      = var.project_id
  display_name = "ML Evaluation Platform - Database"

  dashboard_json = jsonencode({
    displayName = "ML Evaluation Platform - Database"
    mosaicLayout = {
      tiles = [
        # Database Connections
        {
          width  = 6
          height = 4
          widget = {
            title = "Database Connections"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/postgresql/num_backends\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              yAxis = {
                label = "Connections"
                scale = "LINEAR"
              }
            }
          }
        },
        # Database CPU
        {
          xPos   = 6
          width  = 6
          height = 4
          widget = {
            title = "Database CPU Utilization"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/cpu/utilization\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              yAxis = {
                label = "CPU utilization (%)"
                scale = "LINEAR"
              }
            }
          }
        },
        # Database Memory
        {
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Database Memory Usage"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/memory/utilization\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              yAxis = {
                label = "Memory utilization (%)"
                scale = "LINEAR"
              }
            }
          }
        },
        # Database Disk Usage
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Database Disk Usage"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloudsql_database\" AND metric.type=\"cloudsql.googleapis.com/database/disk/utilization\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                  targetAxis = "Y1"
                }
              ]
              yAxis = {
                label = "Disk utilization (%)"
                scale = "LINEAR"
              }
            }
          }
        }
      ]
    }
  })
}

# ML Operations dashboard
resource "google_monitoring_dashboard" "ml_eval_mlops" {
  project      = var.project_id
  display_name = "ML Evaluation Platform - ML Operations"

  dashboard_json = jsonencode({
    displayName = "ML Evaluation Platform - ML Operations"
    mosaicLayout = {
      tiles = [
        # Celery Task Metrics
        {
          width  = 12
          height = 4
          widget = {
            title = "Celery Task Processing"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=~\"ml-eval-celery-.*\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_RATE"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields = ["resource.label.service_name"]
                      }
                    }
                  }
                  plotType = "STACKED_AREA"
                  targetAxis = "Y1"
                }
              ]
              yAxis = {
                label = "Tasks per second"
                scale = "LINEAR"
              }
            }
          }
        },
        # Model Training Duration
        {
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Model Training Duration"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"ml-eval-celery-training\""
                  aggregation = {
                    alignmentPeriod  = "3600s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
              sparkChartView = {
                sparkChartType = "SPARK_LINE"
              }
            }
          }
        },
        # Inference Throughput
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Inference Throughput"
            scorecard = {
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"ml-eval-celery-inference\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
              sparkChartView = {
                sparkChartType = "SPARK_BAR"
              }
            }
          }
        }
      ]
    }
  })
}