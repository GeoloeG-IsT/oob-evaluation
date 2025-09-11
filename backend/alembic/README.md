# Database Migrations for ML Evaluation Platform

This directory contains Alembic database migrations for the ML Evaluation Platform.

## Overview

Alembic is used to manage database schema changes with version control. All migrations are automatically generated and applied using the async SQLAlchemy setup.

## Directory Structure

```
alembic/
├── versions/          # Migration files
├── env.py            # Alembic environment configuration
├── script.py.mako    # Migration template
└── README.md         # This file
```

## Common Commands

All commands should be run from the `backend/` directory:

### Generate a new migration
```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply migrations
```bash
alembic upgrade head
```

### View current migration status
```bash
alembic current
```

### View migration history
```bash
alembic history --verbose
```

### Downgrade to previous migration
```bash
alembic downgrade -1
```

### Downgrade to specific revision
```bash
alembic downgrade <revision_id>
```

## Environment Configuration

The migrations use the following environment variables:

- `DATABASE_URL`: Primary database connection string
- `LOG_LEVEL`: Set to DEBUG for SQL query logging
- `STRUCTURED_LOGGING`: Enable structured logging (true/false)

## Migration Guidelines

1. **Always review generated migrations** before applying them
2. **Test migrations on development data** before production
3. **Use descriptive migration messages** that explain the changes
4. **Backup production data** before running migrations
5. **Plan downgrades** for critical schema changes

## Supabase Integration

The migrations are configured to work with Supabase/PostgreSQL:

- Uses `postgresql+asyncpg://` driver for async operations
- Includes proper connection pooling for migration operations  
- Supports both local PostgreSQL and Supabase cloud instances
- Filters out system tables and schemas during autogenerate

## Troubleshooting

### Connection Issues
- Verify `DATABASE_URL` is correctly set
- Check network connectivity to database
- Ensure database user has migration permissions

### Migration Conflicts
```bash
# View conflicting revisions
alembic branches

# Merge conflicts manually or use
alembic merge -m "Merge conflicting migrations" <rev1> <rev2>
```

### Reset Migration State (Development Only)
```bash
# WARNING: This will lose migration history
alembic stamp head
```

## Integration with Application

The migration system integrates with the application's database connection:

- Uses the same connection configuration as the main application
- Shares SQLAlchemy Base and model definitions
- Includes structured logging integration
- Supports both offline and online migration modes