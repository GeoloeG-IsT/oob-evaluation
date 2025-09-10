# oob-evaluation-claude plan

We are going to use:

- Netx.js/React/Typescript for the frontend
- fastapi/python for the backend (all relevant APIs)
- supabase/postgresql for the database
- celery for the task queue
- docker for the containerization and docker compose for the orchestration
- github actions for the CI/CD
- GCP Cloud Run for the deployment (development)

The project will be a single project with the following structure:

- frontend/
- backend/
- celery/
- docker/
- .env
- .env.local
- .env.development
