"""
Security utilities and best practices for ML Evaluation Platform.

This module provides security validation, key management, and security
best practices implementation across all environments.
"""

import os
import re
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .config import get_settings


logger = logging.getLogger(__name__)


class SecurityManager:
    """Centralized security management for the application."""
    
    def __init__(self):
        """Initialize security manager with settings."""
        self.settings = get_settings()
        self._encryption_key = None
    
    @property
    def encryption_key(self) -> bytes:
        """Get or generate encryption key."""
        if self._encryption_key is None:
            if self.settings.security.encryption_key:
                # Use provided encryption key
                key_data = self.settings.security.encryption_key.encode()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'ml_eval_platform_salt',  # Use a fixed salt for consistency
                    iterations=100000,
                )
                self._encryption_key = base64.urlsafe_b64encode(kdf.derive(key_data))
            else:
                # Generate a new key (development only)
                if self.settings.app.environment == "production":
                    raise ValueError("ENCRYPTION_KEY must be set in production")
                self._encryption_key = Fernet.generate_key()
                logger.warning("Generated ephemeral encryption key for development")
        
        return self._encryption_key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            fernet = Fernet(self.encryption_key)
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash a password with salt.
        
        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        
        hashed = kdf.derive(password.encode())
        return base64.urlsafe_b64encode(hashed).decode(), salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Password to verify
            hashed_password: Stored hash
            salt: Stored salt
            
        Returns:
            True if password matches
        """
        try:
            expected_hash, _ = self.hash_password(password, salt)
            return secrets.compare_digest(expected_hash, hashed_password)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_api_key(self, length: int = 32) -> str:
        """Generate a secure API key.
        
        Args:
            length: Length of the key
            
        Returns:
            Secure API key
        """
        return secrets.token_urlsafe(length)
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if format is valid
        """
        if not api_key or len(api_key) < 16:
            return False
        
        # Check for URL-safe base64 characters
        if not re.match(r'^[A-Za-z0-9_-]+$', api_key):
            return False
        
        return True
    
    def is_admin_api_key(self, api_key: str) -> bool:
        """Check if API key is an admin key.
        
        Args:
            api_key: API key to check
            
        Returns:
            True if it's an admin key
        """
        return api_key in self.settings.security.admin_api_keys
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:250] + ext
        
        # Ensure it doesn't start with a dot
        if sanitized.startswith('.'):
            sanitized = 'file_' + sanitized
        
        return sanitized
    
    def validate_file_upload(self, filename: str, content_type: str, size: int) -> Dict[str, Any]:
        """Validate file upload for security.
        
        Args:
            filename: Original filename
            content_type: MIME content type
            size: File size in bytes
            
        Returns:
            Validation result with details
        """
        errors = []
        warnings = []
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        allowed_extensions = self.settings.storage.upload_allowed_extensions
        
        if file_ext not in allowed_extensions:
            errors.append(f"File extension '{file_ext}' not allowed")
        
        # Check file size
        if size > self.settings.storage.upload_max_size:
            errors.append(f"File size {size} exceeds maximum {self.settings.storage.upload_max_size}")
        
        # Check filename
        sanitized_filename = self.sanitize_filename(filename)
        if sanitized_filename != filename:
            warnings.append(f"Filename sanitized: '{filename}' -> '{sanitized_filename}'")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.\./',  # Directory traversal
            r'[<>:"|?*]',  # Invalid characters
            r'(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)',  # Windows reserved names
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                errors.append(f"Filename contains suspicious pattern: {pattern}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'sanitized_filename': sanitized_filename,
        }


class SecurityValidator:
    """Validates security configuration and practices."""
    
    def __init__(self):
        """Initialize security validator."""
        self.settings = get_settings()
        self.security_manager = SecurityManager()
    
    def validate_security_configuration(self) -> Dict[str, Any]:
        """Validate security configuration.
        
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        info = []
        
        # Validate secret key
        secret_key = self.settings.security.secret_key
        if not secret_key:
            errors.append("SECRET_KEY is not set")
        elif len(secret_key) < 32:
            errors.append("SECRET_KEY should be at least 32 characters long")
        elif self._is_weak_secret(secret_key):
            errors.append("SECRET_KEY appears to be weak or default")
        
        # Validate JWT secret
        if self.settings.security.jwt_secret:
            if len(self.settings.security.jwt_secret) < 32:
                warnings.append("JWT_SECRET should be at least 32 characters long")
        
        # Validate CORS settings
        cors_origins = self.settings.security.cors_origins
        if "*" in cors_origins and self.settings.app.environment == "production":
            warnings.append("CORS origins should be restricted in production")
        
        # Validate HTTPS usage in production
        if self.settings.app.environment == "production":
            backend_url = str(self.settings.app.backend_url)
            frontend_url = str(self.settings.app.frontend_url)
            
            if not backend_url.startswith("https://"):
                warnings.append("Backend URL should use HTTPS in production")
            
            if not frontend_url.startswith("https://"):
                warnings.append("Frontend URL should use HTTPS in production")
        
        # Validate rate limiting
        if not self.settings.security.rate_limit_enabled:
            warnings.append("Rate limiting is disabled")
        
        # Validate API keys
        for api_key in self.settings.security.admin_api_keys:
            if not self.security_manager.validate_api_key(api_key):
                errors.append(f"Invalid admin API key format: {api_key[:8]}...")
        
        # Check environment-specific requirements
        if self.settings.app.environment == "production":
            self._validate_production_security(errors, warnings, info)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'info': info,
        }
    
    def _is_weak_secret(self, secret: str) -> bool:
        """Check if secret appears to be weak.
        
        Args:
            secret: Secret to check
            
        Returns:
            True if secret is weak
        """
        weak_secrets = [
            "your-secret-key-here",
            "development-secret-key",
            "test-secret-key",
            "changeme",
            "secret",
            "password",
            "123456",
            "admin",
        ]
        
        return secret.lower() in [s.lower() for s in weak_secrets]
    
    def _validate_production_security(self, errors: List[str], warnings: List[str], info: List[str]):
        """Validate production-specific security requirements.
        
        Args:
            errors: List to append errors to
            warnings: List to append warnings to
            info: List to append info to
        """
        # Debug mode should be disabled
        if self.settings.app.debug:
            errors.append("DEBUG mode must be disabled in production")
        
        # Secret Manager should be used
        if not self.settings.gcp.use_secret_manager:
            warnings.append("GCP Secret Manager should be used in production")
        
        # GCS should be used for storage
        if not self.settings.storage.use_gcs:
            warnings.append("Google Cloud Storage should be used in production")
        
        # Encryption key should be set
        if not self.settings.security.encryption_key:
            warnings.append("ENCRYPTION_KEY should be set in production")
        
        # Database SSL should be used
        db_url = str(self.settings.database.database_url)
        if "sslmode" not in db_url and not db_url.startswith("sqlite"):
            warnings.append("Database should use SSL in production")
        
        info.append("Production security validation completed")
    
    def generate_security_report(self) -> str:
        """Generate a comprehensive security report.
        
        Returns:
            Security report as string
        """
        validation_result = self.validate_security_configuration()
        
        report = []
        report.append("ML Evaluation Platform - Security Report")
        report.append("=" * 50)
        report.append(f"Environment: {self.settings.app.environment}")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary
        status = "✅ PASS" if validation_result['valid'] else "❌ FAIL"
        report.append(f"Overall Status: {status}")
        report.append(f"Errors: {len(validation_result['errors'])}")
        report.append(f"Warnings: {len(validation_result['warnings'])}")
        report.append("")
        
        # Errors
        if validation_result['errors']:
            report.append("ERRORS:")
            for error in validation_result['errors']:
                report.append(f"  ❌ {error}")
            report.append("")
        
        # Warnings
        if validation_result['warnings']:
            report.append("WARNINGS:")
            for warning in validation_result['warnings']:
                report.append(f"  ⚠️  {warning}")
            report.append("")
        
        # Security checklist
        report.append("SECURITY CHECKLIST:")
        checklist = self._get_security_checklist()
        for item, status in checklist.items():
            status_icon = "✅" if status else "❌"
            report.append(f"  {status_icon} {item}")
        
        return "\n".join(report)
    
    def _get_security_checklist(self) -> Dict[str, bool]:
        """Get security checklist with status.
        
        Returns:
            Dictionary of checklist items and their status
        """
        settings = self.settings
        
        checklist = {
            "Strong secret key (32+ chars)": len(settings.security.secret_key) >= 32,
            "Non-default secret key": not self._is_weak_secret(settings.security.secret_key),
            "HTTPS URLs in production": (
                settings.app.environment != "production" or 
                (str(settings.app.backend_url).startswith("https://") and 
                 str(settings.app.frontend_url).startswith("https://"))
            ),
            "Debug disabled in production": (
                settings.app.environment != "production" or not settings.app.debug
            ),
            "Rate limiting enabled": settings.security.rate_limit_enabled,
            "CORS origins restricted": "*" not in settings.security.cors_origins,
            "Encryption key set": bool(settings.security.encryption_key),
            "Secret Manager in production": (
                settings.app.environment != "production" or settings.gcp.use_secret_manager
            ),
            "Cloud storage in production": (
                settings.app.environment != "production" or settings.storage.use_gcs
            ),
            "Database SSL configured": (
                "sslmode" in str(settings.database.database_url) or 
                str(settings.database.database_url).startswith("sqlite")
            ),
        }
        
        return checklist


def validate_request_security(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate request data for security issues.
    
    Args:
        request_data: Request data to validate
        
    Returns:
        Validation results
    """
    errors = []
    warnings = []
    
    # Check for SQL injection patterns
    sql_patterns = [
        r"(union\s+select)",
        r"(drop\s+table)",
        r"(delete\s+from)",
        r"(insert\s+into)",
        r"(update\s+set)",
        r"('|(\\)|(--)|(;))",
    ]
    
    # Check for XSS patterns
    xss_patterns = [
        r"<script",
        r"javascript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
    ]
    
    # Validate string values
    for key, value in request_data.items():
        if isinstance(value, str):
            # Check SQL injection
            for pattern in sql_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    errors.append(f"Potential SQL injection in field '{key}'")
                    break
            
            # Check XSS
            for pattern in xss_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    errors.append(f"Potential XSS in field '{key}'")
                    break
            
            # Check for extremely long values
            if len(value) > 10000:
                warnings.append(f"Very long value in field '{key}' ({len(value)} chars)")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
    }


def get_security_headers() -> Dict[str, str]:
    """Get recommended security headers.
    
    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }


# Global security manager instance
security_manager = SecurityManager()
security_validator = SecurityValidator()


if __name__ == "__main__":
    # CLI for security validation
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            result = security_validator.validate_security_configuration()
            print("Security Validation Results:")
            print(f"Valid: {result['valid']}")
            if result['errors']:
                print("Errors:")
                for error in result['errors']:
                    print(f"  - {error}")
            if result['warnings']:
                print("Warnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
            sys.exit(0 if result['valid'] else 1)
        
        elif command == "report":
            report = security_validator.generate_security_report()
            print(report)
        
        elif command == "generate-key":
            key = security_manager.generate_api_key()
            print(f"Generated API key: {key}")
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print("Usage: python security.py [validate|report|generate-key]")
        sys.exit(1)