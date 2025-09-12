# Implementation Guide: ML Evaluation Platform

## Overview

This guide provides a complete roadmap for implementing the ML Evaluation Platform, with clear references between design documents and implementation tasks.

## Document Hierarchy and Usage

### 1. Foundation Documents
- **spec.md**: 40 functional requirements defining what to build
- **plan.md**: Architecture decisions and implementation strategy
- **research.md**: Technology choices and best practices

### 2. Design Documents
- **data-model.md**: 8 entities with field definitions and relationships
- **contracts/api-spec.yaml**: 15 API endpoints with complete OpenAPI specification
- **quickstart.md**: 8 validation workflows with expected outcomes

### 3. Implementation Bridge
- **task-mapping.md**: Detailed cross-references between design docs and tasks
- **implementation-guide.md**: This document - complete implementation roadmap

## Implementation Sequence

### Phase 1: Environment Setup
**Reference**: task-mapping.md "Technology Stack Task Breakdown"

**Tasks**:
1. Initialize project structure per plan.md (frontend/, backend/, celery/, docker/)
2. Setup development environment with Next.js 14, FastAPI, PostgreSQL
3. Configure Docker containers for each service
4. Setup CI/CD pipeline with GitHub Actions

**Validation**: All services start successfully with `docker-compose up`

### Phase 2: Database Foundation
**Reference**: data-model.md entities → task-mapping.md "Database Models"

**Tasks**:
1. Create 8 migration files (001-008) for each entity
2. Implement 8 Pydantic model classes in backend/src/models/
3. Setup Alembic migrations and database schema
4. Create database indexes and constraints per data-model.md

**Validation**: All entities can be created, queried, and relationships work

### Phase 3: Contract Tests (TDD Phase)
**Reference**: contracts/api-spec.yaml → task-mapping.md "Contract Test Task Mapping"

**CRITICAL**: These tests MUST be written and MUST FAIL before implementation

**Tasks**:
1. Write 15 contract test files in backend/tests/contract/
2. Each test validates request/response schemas from api-spec.yaml
3. Tests must fail initially (no implementation exists)
4. Verify all HTTP status codes and error responses

**Validation**: All contract tests fail with clear error messages

### Phase 4: Integration Tests
**Reference**: quickstart.md workflows → task-mapping.md "Integration Test Task Mapping"

**Tasks**:
1. Write 8 integration test files covering complete user workflows
2. Tests validate end-to-end functionality per quickstart scenarios
3. Include API validation examples from quickstart.md
4. Tests must fail initially (no services implemented)

**Validation**: All integration tests fail but with clear workflow definitions

### Phase 5: ML Framework Integration
**Reference**: research.md ML decisions → task-mapping.md "ML Integration Task Breakdown"

**Tasks**:
1. Create ml-models library with YOLO11/12, RT-DETR, SAM2 wrappers
2. Implement inference-engine library for real-time and batch processing
3. Create training-pipeline library for model fine-tuning
4. Implement annotation-tools library for assisted annotation

**Validation**: All model frameworks load and can process sample images

### Phase 6: API Implementation
**Reference**: contracts/api-spec.yaml operations → task-mapping.md "API Services"

**Tasks**:
1. Implement 8 service classes in backend/src/services/
2. Create 15 API endpoint implementations in backend/src/api/v1/
3. Implement request validation and error handling
4. Add structured logging for all operations

**Validation**: Contract tests start passing one by one

### Phase 7: Async Processing
**Reference**: Celery setup → task-mapping.md "Celery Setup"

**Tasks**:
1. Implement 4 Celery worker types for long-running operations
2. Create task monitoring and progress tracking
3. Implement result storage and retrieval
4. Add error handling and retry logic

**Validation**: Training and batch inference jobs execute successfully

### Phase 8: Frontend Components
**Reference**: task-mapping.md "Frontend Components"

**Tasks**:
1. Create 7 React component groups for each major feature
2. Implement real-time progress monitoring with WebSockets
3. Create annotation drawing tools with Canvas API
4. Add model deployment dashboard

**Validation**: All quickstart workflows work through web interface

### Phase 9: CLI Implementation
**Reference**: Constitutional requirements → task-mapping.md "CLI Commands"

**Tasks**:
1. Create 4 CLI command modules for each library
2. Implement --help, --version, --format flags for each command
3. Add library documentation in llms.txt format
4. Ensure each feature is accessible via CLI

**Validation**: All features available through command line

### Phase 10: Deployment and Validation
**Reference**: quickstart.md validation → plan.md Phase 5

**Tasks**:
1. Create production Docker configurations
2. Setup GCP Cloud Run deployment
3. Execute complete quickstart validation
4. Performance testing and optimization

**Validation**: Complete quickstart guide passes end-to-end

## Cross-Reference Quick Guide

### Finding Implementation Details

**For API Endpoints**:
1. Find endpoint in contracts/api-spec.yaml
2. Look up operationId in task-mapping.md "Contract Test Task Mapping"
3. Find service implementation in task-mapping.md "API Services"

**For Data Operations**:
1. Find entity in data-model.md
2. Look up model file in task-mapping.md "Database Models" 
3. Find related service in task-mapping.md "API Services"

**For User Workflows**:
1. Find workflow in quickstart.md
2. Look up integration test in task-mapping.md "Integration Test Task Mapping"
3. Find frontend component in task-mapping.md "Frontend Components"

**For ML Operations**:
1. Find requirement in spec.md (FR-016, FR-017 for models)
2. Look up framework in research.md decisions
3. Find implementation in task-mapping.md "ML Integration Task Breakdown"

## Key Implementation Principles

### Constitutional Compliance
- **TDD Enforced**: Contract tests → Integration tests → Implementation
- **Library Architecture**: Every feature as library with CLI
- **Real Dependencies**: Actual PostgreSQL, no mocks in integration tests
- **Versioning**: 1.0.0 with BUILD increments

### File Organization
- **Backend**: backend/src/{models,services,api,lib,cli}/
- **Frontend**: frontend/src/{components,pages,services}/
- **Tests**: backend/tests/{contract,integration,unit}/
- **Docs**: specs/001-oob-evaluation-claude/

### Quality Gates
1. All contract tests must fail before implementation
2. Implementation only to make tests pass
3. Integration tests validate complete workflows
4. Quickstart guide must execute successfully
5. All ML models must load and process test images

## Success Criteria

✅ **Complete Implementation**: All 50-60 tasks executed following TDD principles

✅ **Functional Validation**: Complete quickstart guide executes successfully

✅ **Performance Requirements**: 
- Real-time inference monitoring
- Batch processing with progress tracking
- Unlimited file size handling

✅ **Constitutional Compliance**:
- Library architecture with CLI interfaces
- TDD workflow with failing tests first
- Real dependencies in integration tests

✅ **Documentation**: 
- API documentation auto-generated from OpenAPI spec
- Library documentation in llms.txt format
- Agent context maintained in CLAUDE.md

This implementation guide ensures clear traceability from requirements through design to implementation, with explicit cross-references enabling efficient task execution.