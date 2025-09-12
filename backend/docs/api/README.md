# ML Evaluation Platform API

API for object detection and segmentation model evaluation platform

**Version:** 1.0.0  
**Generated:** 2025-09-11 17:24:19

## Table of Contents

- [POST /api/v1/images](#uploadimages) - Upload image files\n- [GET /api/v1/images](#listimages) - List images\n- [GET /api/v1/images/{image_id}](#getimage) - Get image details\n- [POST /api/v1/annotations](#createannotation) - Create manual annotation\n- [GET /api/v1/annotations](#listannotations) - List annotations\n- [POST /api/v1/annotations/assisted](#generateassistedannotation) - Generate assisted annotation\n- [GET /api/v1/models](#listmodels) - List available models\n- [GET /api/v1/models/{model_id}](#getmodel) - Get model details\n- [POST /api/v1/inference/single](#runsingleinference) - Run single image inference\n- [POST /api/v1/inference/batch](#runbatchinference) - Run batch inference\n- [GET /api/v1/inference/jobs/{job_id}](#getinferencejob) - Get inference job status\n- [POST /api/v1/training/jobs](#starttraining) - Start model training\n- [GET /api/v1/training/jobs/{job_id}](#gettrainingjob) - Get training job status\n- [POST /api/v1/evaluation/metrics](#calculatemetrics) - Calculate performance metrics\n- [POST /api/v1/evaluation/compare](#comparemodels) - Compare model performance\n- [POST /api/v1/deployments](#deploymodel) - Deploy model\n- [GET /api/v1/deployments](#listdeployments) - List deployments\n- [GET /api/v1/deployments/{deployment_id}](#getdeployment) - Get deployment details\n- [PATCH /api/v1/deployments/{deployment_id}](#updatedeployment) - Update deployment\n- [POST /api/v1/export/annotations](#exportannotations) - Export annotations\n\n## Endpoints\n\n### POST /api/v1/images

**Upload image files**

Upload images to the server for processing and evaluation

**Operation ID:** `uploadImages`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `multipart/form-data`\n\n```json\n{
  "type": "object",
  "properties": {
    "files": {
      "type": "array",
      "items": {
        "type": "string",
        "format": "binary"
      }
    },
    "dataset_split": {
      "type": "string",
      "enum": [
        "train",
        "validation",
        "test"
      ],
      "default": "train"
    }
  },
  "required": [
    "files"
  ]
}\n```\n\n#### Responses\n\n**Status 201**\n\nImages uploaded successfully\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/ImageUploadResponse"
}\n```\n\n**Status 400**\n\nNo description\n\n**Status 413**\n\nNo description\n\n---\n\n### GET /api/v1/images

**List images**

Retrieve list of uploaded images with filtering

**Operation ID:** `listImages`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `dataset_split` | query | string | No | No description |\n| `limit` | query | integer | No | No description |\n| `offset` | query | integer | No | No description |\n#### Responses\n\n**Status 200**\n\nList of images\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/ImageListResponse"
}\n```\n\n---\n\n### GET /api/v1/images/{image_id}

**Get image details**

Retrieve detailed information about a specific image

**Operation ID:** `getImage`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `image_id` | path | string | Yes | No description |\n#### Responses\n\n**Status 200**\n\nImage details\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/Image"
}\n```\n\n**Status 404**\n\nNo description\n\n---\n\n### POST /api/v1/annotations

**Create manual annotation**

Create user annotation for an image

**Operation ID:** `createAnnotation`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/CreateAnnotationRequest"
}\n```\n\n#### Responses\n\n**Status 201**\n\nAnnotation created\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/Annotation"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n### GET /api/v1/annotations

**List annotations**

Retrieve annotations with filtering options

**Operation ID:** `listAnnotations`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `image_id` | query | string | No | No description |\n| `model_id` | query | string | No | No description |\n| `creation_method` | query | string | No | No description |\n#### Responses\n\n**Status 200**\n\nList of annotations\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/AnnotationListResponse"
}\n```\n\n---\n\n### POST /api/v1/annotations/assisted

**Generate assisted annotation**

Use pre-trained model to generate annotation suggestions

**Operation ID:** `generateAssistedAnnotation`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/AssistedAnnotationRequest"
}\n```\n\n#### Responses\n\n**Status 201**\n\nAssisted annotation generated\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/Annotation"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n### GET /api/v1/models

**List available models**

Retrieve list of available models for inference and training

**Operation ID:** `listModels`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `type` | query | string | No | No description |\n| `framework` | query | string | No | No description |\n#### Responses\n\n**Status 200**\n\nList of models\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/ModelListResponse"
}\n```\n\n---\n\n### GET /api/v1/models/{model_id}

**Get model details**

Retrieve detailed information about a specific model

**Operation ID:** `getModel`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `model_id` | path | string | Yes | No description |\n#### Responses\n\n**Status 200**\n\nModel details\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/Model"
}\n```\n\n**Status 404**\n\nNo description\n\n---\n\n### POST /api/v1/inference/single

**Run single image inference**

Run model inference on a single image

**Operation ID:** `runSingleInference`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/SingleInferenceRequest"
}\n```\n\n#### Responses\n\n**Status 200**\n\nInference completed\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/InferenceResult"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n### POST /api/v1/inference/batch

**Run batch inference**

Run model inference on multiple images

**Operation ID:** `runBatchInference`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/BatchInferenceRequest"
}\n```\n\n#### Responses\n\n**Status 202**\n\nBatch inference job created\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/InferenceJob"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n### GET /api/v1/inference/jobs/{job_id}

**Get inference job status**

Monitor batch inference job progress

**Operation ID:** `getInferenceJob`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `job_id` | path | string | Yes | No description |\n#### Responses\n\n**Status 200**\n\nJob status\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/InferenceJob"
}\n```\n\n**Status 404**\n\nNo description\n\n---\n\n### POST /api/v1/training/jobs

**Start model training**

Initiate fine-tuning of a model with user annotations

**Operation ID:** `startTraining`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/TrainingRequest"
}\n```\n\n#### Responses\n\n**Status 202**\n\nTraining job created\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/TrainingJob"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n### GET /api/v1/training/jobs/{job_id}

**Get training job status**

Monitor training job progress and logs

**Operation ID:** `getTrainingJob`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `job_id` | path | string | Yes | No description |\n#### Responses\n\n**Status 200**\n\nTraining job status\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/TrainingJob"
}\n```\n\n**Status 404**\n\nNo description\n\n---\n\n### POST /api/v1/evaluation/metrics

**Calculate performance metrics**

Evaluate model performance on test data

**Operation ID:** `calculateMetrics`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/MetricsRequest"
}\n```\n\n#### Responses\n\n**Status 200**\n\nPerformance metrics calculated\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/MetricsResponse"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n### POST /api/v1/evaluation/compare

**Compare model performance**

Compare performance metrics between multiple models

**Operation ID:** `compareModels`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/ModelComparisonRequest"
}\n```\n\n#### Responses\n\n**Status 200**\n\nModel comparison results\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/ModelComparisonResponse"
}\n```\n\n---\n\n### POST /api/v1/deployments

**Deploy model**

Deploy trained model as REST API endpoint

**Operation ID:** `deployModel`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/DeploymentRequest"
}\n```\n\n#### Responses\n\n**Status 202**\n\nDeployment initiated\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/Deployment"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n### GET /api/v1/deployments

**List deployments**

Retrieve list of deployed models

**Operation ID:** `listDeployments`

#### Parameters

No parameters\n#### Responses\n\n**Status 200**\n\nList of deployments\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/DeploymentListResponse"
}\n```\n\n---\n\n### GET /api/v1/deployments/{deployment_id}

**Get deployment details**

Retrieve deployment status and metrics

**Operation ID:** `getDeployment`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `deployment_id` | path | string | Yes | No description |\n#### Responses\n\n**Status 200**\n\nDeployment details\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/Deployment"
}\n```\n\n**Status 404**\n\nNo description\n\n---\n\n### PATCH /api/v1/deployments/{deployment_id}

**Update deployment**

Update deployment configuration or status

**Operation ID:** `updateDeployment`

#### Parameters

| Name | In | Type | Required | Description |\n|------|----|----- |----------|-------------|\n| `deployment_id` | path | string | Yes | No description |\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/DeploymentUpdateRequest"
}\n```\n\n#### Responses\n\n**Status 200**\n\nDeployment updated\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/Deployment"
}\n```\n\n---\n\n### POST /api/v1/export/annotations

**Export annotations**

Export annotations in standard formats

**Operation ID:** `exportAnnotations`

#### Parameters

No parameters\n\n#### Request Body\n\n**Content-Type:** `application/json`\n\n```json\n{
  "$ref": "#/components/schemas/ExportRequest"
}\n```\n\n#### Responses\n\n**Status 200**\n\nExport file\n\n**Content-Type:** `application/zip`\n\n```json\n{
  "type": "string",
  "format": "binary"
}\n```\n\n**Status 400**\n\nNo description\n\n---\n\n