"""
Error handling utilities for secure error responses and logging
"""
import logging
import traceback
from typing import Dict, Any, Optional, TYPE_CHECKING
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime, timezone
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastapi import Request as RequestType
else:
    RequestType = object


class ErrorCategories:
    """Error categories for consistent error handling"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
    CONFLICT_ERROR = "CONFLICT_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"


class SecureErrorResponse:
    """Generate secure error responses that don't leak sensitive information"""
    
    @staticmethod
    def generate_error_id() -> str:
        """Generate unique error ID for tracking"""
        return str(uuid.uuid4())[:8]
    
    @staticmethod
    def log_detailed_error(
        error_id: str,
        error: Exception,
        request: Optional["RequestType"] = None,
        user_id: Optional[int] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Log detailed error information server-side, including request body and query params if available"""
        error_details = {
            "error_id": error_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "user_id": user_id,
            "additional_context": additional_context or {}
        }
        
        if request:
            error_details.update({
                "request_method": getattr(request, 'method', None),
                "request_url": str(getattr(request, 'url', '')),
                "request_headers": dict(getattr(request, 'headers', {})),
                "client_ip": getattr(getattr(request, 'client', None), 'host', 'unknown') if getattr(request, 'client', None) else 'unknown'
            })
            # Try to log query params
            try:
                error_details["query_params"] = dict(getattr(request, 'query_params', {}))
            except Exception:
                error_details["query_params"] = "<unavailable>"
            # Try to log request body for POST/PUT/PATCH
            if getattr(request, 'method', None) in ("POST", "PUT", "PATCH"):
                try:
                    import asyncio
                    body = None
                    if hasattr(request, "_body") and request._body is not None:
                        body = request._body
                    elif hasattr(request, "body"):
                        body = asyncio.run(request.body())
                    if body is not None:
                        import json
                        try:
                            error_details["request_body"] = json.loads(body.decode("utf-8"))
                        except Exception:
                            error_details["request_body"] = body.decode("utf-8", errors="replace")
                except Exception:
                    error_details["request_body"] = "<unavailable>"
        # Write detailed error info to app_errors.log
        try:
            with open('app_errors.log', 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(error_details, ensure_ascii=False, indent=2) + '\n')
        except Exception as log_exc:
            logger.error(f"Failed to write detailed error to app_errors.log: {log_exc}")
        logger.error(f"Detailed error logged: {error_details}")
    
    @staticmethod
    def validation_error(
        detail: str = "Invalid input data",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic validation error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=422,
            content={
                "error": ErrorCategories.VALIDATION_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )
    
    @staticmethod
    def authentication_error(
        detail: str = "Authentication failed",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic authentication error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=401,
            content={
                "error": ErrorCategories.AUTHENTICATION_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )
    
    @staticmethod
    def authorization_error(
        detail: str = "Access denied",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic authorization error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=403,
            content={
                "error": ErrorCategories.AUTHORIZATION_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )
    
    @staticmethod
    def not_found_error(
        detail: str = "Resource not found",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic not found error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=404,
            content={
                "error": ErrorCategories.NOT_FOUND_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )
    
    @staticmethod
    def conflict_error(
        detail: str = "Resource conflict",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic conflict error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=409,
            content={
                "error": ErrorCategories.CONFLICT_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )
    
    @staticmethod
    def database_error(
        detail: str = "Database operation failed",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic database error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": ErrorCategories.DATABASE_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )
    
    @staticmethod
    def internal_server_error(
        detail: str = "An internal server error occurred",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic internal server error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": ErrorCategories.INTERNAL_SERVER_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )
    
    @staticmethod
    def rate_limit_error(
        detail: str = "Too many requests",
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Return generic rate limit error response"""
        if not error_id:
            error_id = SecureErrorResponse.generate_error_id()
        
        return JSONResponse(
            status_code=429,
            content={
                "error": ErrorCategories.RATE_LIMIT_ERROR,
                "message": detail,
                "error_id": error_id
            }
        )


def handle_database_error(error: SQLAlchemyError, request: Optional["RequestType"] = None, user_id: Optional[int] = None) -> JSONResponse:
    """Handle database errors securely"""
    error_id = SecureErrorResponse.generate_error_id()
    
    # Log detailed error information
    SecureErrorResponse.log_detailed_error(
        error_id=error_id,
        error=error,
        request=request,
        user_id=user_id,
        additional_context={"error_category": "database"}
    )
    
    # Determine specific error type for generic response
    if isinstance(error, IntegrityError):
        if "UNIQUE constraint failed" in str(error):
            return SecureErrorResponse.conflict_error("Resource already exists", error_id)
        elif "FOREIGN KEY constraint failed" in str(error):
            return SecureErrorResponse.validation_error("Invalid reference data", error_id)
        else:
            return SecureErrorResponse.database_error("Data integrity error", error_id)
    
    # For all other database errors, return generic message
    return SecureErrorResponse.database_error("Database operation failed", error_id)


def handle_validation_error(error: ValidationError, request: Optional["RequestType"] = None, user_id: Optional[int] = None) -> JSONResponse:
    """Handle Pydantic validation errors securely"""
    error_id = SecureErrorResponse.generate_error_id()
    
    # Log detailed validation error
    SecureErrorResponse.log_detailed_error(
        error_id=error_id,
        error=error,
        request=request,
        user_id=user_id,
        additional_context={
            "error_category": "validation",
            "validation_errors": error.errors()
        }
    )
    
    # Return generic validation error message
    return SecureErrorResponse.validation_error("Invalid input data provided", error_id)


def handle_generic_exception(error: Exception, request: Optional["RequestType"] = None, user_id: Optional[int] = None) -> JSONResponse:
    """Handle any unexpected exceptions securely"""
    error_id = SecureErrorResponse.generate_error_id()
    
    # Log detailed error information
    SecureErrorResponse.log_detailed_error(
        error_id=error_id,
        error=error,
        request=request,
        user_id=user_id,
        additional_context={"error_category": "unexpected"}
    )
    
    # Return generic internal server error
    return SecureErrorResponse.internal_server_error("An internal server error occurred", error_id)


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages to remove sensitive information"""
    # List of sensitive patterns to remove
    sensitive_patterns = [
        r'password[=:]\s*\S+',
        r'token[=:]\s*\S+',
        r'secret[=:]\s*\S+',
        r'key[=:]\s*\S+',
        r'user\s+\'[^\']+\'',
        r'database.*connection.*failed.*user.*',
        r'permission.*denied.*user.*',
        r'authentication.*failed.*user.*',
        r'file.*not.*found.*[\/\\].*',
        r'path.*[\/\\].*not.*found',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # email addresses
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
    ]
    
    import re
    sanitized = message
    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
    
    return sanitized
