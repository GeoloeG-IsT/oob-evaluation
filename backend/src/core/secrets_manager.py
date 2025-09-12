"""
GCP Secret Manager integration for secure secrets management.

This module provides integration with Google Cloud Secret Manager for
storing and retrieving sensitive configuration data in production environments.
"""

import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from google.cloud import secretmanager
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manages secrets using GCP Secret Manager."""
    
    def __init__(self, project_id: Optional[str] = None):
        """Initialize Secret Manager client.
        
        Args:
            project_id: GCP project ID. If not provided, uses environment variable.
        """
        if not GCP_AVAILABLE:
            raise ImportError("Google Cloud Secret Manager not available. Install google-cloud-secret-manager")
        
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID must be provided or set as environment variable")
        
        try:
            self.client = secretmanager.SecretManagerServiceClient()
            logger.info(f"Secret Manager client initialized for project: {self.project_id}")
        except DefaultCredentialsError:
            logger.error("GCP credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or use gcloud auth")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Secret Manager client: {e}")
            raise
    
    def get_secret(self, secret_name: str, version: str = "latest") -> Optional[str]:
        """Retrieve a secret from Secret Manager.
        
        Args:
            secret_name: Name of the secret
            version: Version of the secret (default: "latest")
            
        Returns:
            Secret value as string, or None if not found
        """
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            logger.debug(f"Retrieved secret: {secret_name}")
            return secret_value
        except Exception as e:
            logger.warning(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    def create_secret(self, secret_name: str, secret_value: str) -> bool:
        """Create a new secret in Secret Manager.
        
        Args:
            secret_name: Name of the secret
            secret_value: Value to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            parent = f"projects/{self.project_id}"
            
            # Create the secret
            secret = self.client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_name,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            
            # Add the secret version
            self.client.add_secret_version(
                request={
                    "parent": secret.name,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            logger.info(f"Created secret: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create secret {secret_name}: {e}")
            return False
    
    def update_secret(self, secret_name: str, secret_value: str) -> bool:
        """Update an existing secret in Secret Manager.
        
        Args:
            secret_name: Name of the secret
            secret_value: New value to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            parent = f"projects/{self.project_id}/secrets/{secret_name}"
            
            # Add new version to existing secret
            self.client.add_secret_version(
                request={
                    "parent": parent,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            logger.info(f"Updated secret: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update secret {secret_name}: {e}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from Secret Manager.
        
        Args:
            secret_name: Name of the secret to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}"
            self.client.delete_secret(request={"name": name})
            logger.info(f"Deleted secret: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_name}: {e}")
            return False
    
    def list_secrets(self) -> Dict[str, Any]:
        """List all secrets in the project.
        
        Returns:
            Dictionary with secret names and metadata
        """
        try:
            parent = f"projects/{self.project_id}"
            secrets = {}
            
            for secret in self.client.list_secrets(request={"parent": parent}):
                secret_name = secret.name.split("/")[-1]
                secrets[secret_name] = {
                    "name": secret.name,
                    "create_time": secret.create_time,
                    "labels": dict(secret.labels) if secret.labels else {},
                }
            
            logger.info(f"Listed {len(secrets)} secrets")
            return secrets
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return {}


def get_secret_from_gcp(secret_name: str, project_id: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret from GCP Secret Manager.
    
    Args:
        secret_name: Name of the secret
        project_id: GCP project ID (optional)
        
    Returns:
        Secret value or None if not found
    """
    try:
        manager = SecretsManager(project_id)
        return manager.get_secret(secret_name)
    except Exception as e:
        logger.error(f"Error getting secret {secret_name}: {e}")
        return None


class LocalSecretsManager:
    """Local secrets manager for development using .env files."""
    
    def __init__(self, secrets_file: str = ".env.secrets"):
        """Initialize local secrets manager.
        
        Args:
            secrets_file: Path to the secrets file
        """
        self.secrets_file = Path(secrets_file)
        self.secrets: Dict[str, str] = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from file."""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            self.secrets[key.strip()] = value.strip()
                logger.info(f"Loaded {len(self.secrets)} secrets from {self.secrets_file}")
            except Exception as e:
                logger.error(f"Failed to load secrets from {self.secrets_file}: {e}")
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret from local storage.
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Secret value or None if not found
        """
        return self.secrets.get(secret_name)
    
    def set_secret(self, secret_name: str, secret_value: str) -> bool:
        """Set a secret in local storage.
        
        Args:
            secret_name: Name of the secret
            secret_value: Value to store
            
        Returns:
            True if successful
        """
        try:
            self.secrets[secret_name] = secret_value
            self._save_secrets()
            logger.info(f"Set local secret: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {secret_name}: {e}")
            return False
    
    def _save_secrets(self):
        """Save secrets to file."""
        try:
            with open(self.secrets_file, 'w') as f:
                f.write("# Local secrets file - DO NOT COMMIT TO VERSION CONTROL\n")
                f.write("# Generated by ML Evaluation Platform\n\n")
                for key, value in sorted(self.secrets.items()):
                    f.write(f"{key}={value}\n")
        except Exception as e:
            logger.error(f"Failed to save secrets to {self.secrets_file}: {e}")


def migrate_secrets_to_gcp(
    local_secrets_file: str = ".env.secrets",
    project_id: Optional[str] = None,
    dry_run: bool = True
) -> Dict[str, bool]:
    """Migrate secrets from local file to GCP Secret Manager.
    
    Args:
        local_secrets_file: Path to local secrets file
        project_id: GCP project ID
        dry_run: If True, only show what would be migrated
        
    Returns:
        Dictionary with migration results
    """
    local_manager = LocalSecretsManager(local_secrets_file)
    results = {}
    
    if not local_manager.secrets:
        logger.warning("No local secrets found to migrate")
        return results
    
    if dry_run:
        logger.info("DRY RUN: The following secrets would be migrated:")
        for secret_name in local_manager.secrets:
            logger.info(f"  - {secret_name}")
            results[secret_name] = True
        return results
    
    try:
        gcp_manager = SecretsManager(project_id)
        
        for secret_name, secret_value in local_manager.secrets.items():
            try:
                success = gcp_manager.create_secret(secret_name, secret_value)
                results[secret_name] = success
                
                if success:
                    logger.info(f"Migrated secret: {secret_name}")
                else:
                    logger.error(f"Failed to migrate secret: {secret_name}")
            except Exception as e:
                logger.error(f"Error migrating secret {secret_name}: {e}")
                results[secret_name] = False
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Migration completed: {success_count}/{len(results)} secrets migrated")
        
    except Exception as e:
        logger.error(f"Failed to initialize GCP Secret Manager: {e}")
        for secret_name in local_manager.secrets:
            results[secret_name] = False
    
    return results


if __name__ == "__main__":
    # CLI for secrets management
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python secrets_manager.py [command] [args...]")
        print("Commands:")
        print("  get <secret_name> [project_id]")
        print("  create <secret_name> <secret_value> [project_id]")
        print("  update <secret_name> <secret_value> [project_id]")
        print("  delete <secret_name> [project_id]")
        print("  list [project_id]")
        print("  migrate [local_file] [project_id] [--dry-run]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "get":
            secret_name = sys.argv[2]
            project_id = sys.argv[3] if len(sys.argv) > 3 else None
            value = get_secret_from_gcp(secret_name, project_id)
            if value:
                print(value)
            else:
                print(f"Secret '{secret_name}' not found")
                sys.exit(1)
        
        elif command == "create":
            secret_name = sys.argv[2]
            secret_value = sys.argv[3]
            project_id = sys.argv[4] if len(sys.argv) > 4 else None
            manager = SecretsManager(project_id)
            if manager.create_secret(secret_name, secret_value):
                print(f"Secret '{secret_name}' created successfully")
            else:
                print(f"Failed to create secret '{secret_name}'")
                sys.exit(1)
        
        elif command == "update":
            secret_name = sys.argv[2]
            secret_value = sys.argv[3]
            project_id = sys.argv[4] if len(sys.argv) > 4 else None
            manager = SecretsManager(project_id)
            if manager.update_secret(secret_name, secret_value):
                print(f"Secret '{secret_name}' updated successfully")
            else:
                print(f"Failed to update secret '{secret_name}'")
                sys.exit(1)
        
        elif command == "delete":
            secret_name = sys.argv[2]
            project_id = sys.argv[3] if len(sys.argv) > 3 else None
            manager = SecretsManager(project_id)
            if manager.delete_secret(secret_name):
                print(f"Secret '{secret_name}' deleted successfully")
            else:
                print(f"Failed to delete secret '{secret_name}'")
                sys.exit(1)
        
        elif command == "list":
            project_id = sys.argv[2] if len(sys.argv) > 2 else None
            manager = SecretsManager(project_id)
            secrets = manager.list_secrets()
            print(json.dumps(secrets, indent=2, default=str))
        
        elif command == "migrate":
            local_file = sys.argv[2] if len(sys.argv) > 2 else ".env.secrets"
            project_id = sys.argv[3] if len(sys.argv) > 3 else None
            dry_run = "--dry-run" in sys.argv
            
            results = migrate_secrets_to_gcp(local_file, project_id, dry_run)
            print(json.dumps(results, indent=2))
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)