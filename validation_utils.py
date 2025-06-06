"""
Validation utilities for comprehensive input validation and sanitization
"""
import re
import html
import bleach
from typing import Optional, List, Any
from datetime import datetime, date, time, timezone
from pydantic import validator, ValidationError


class ValidationConstants:
    """Constants for validation rules"""
    
    # String length limits
    MIN_NAME_LENGTH = 1
    MAX_NAME_LENGTH = 100
    MIN_USERNAME_LENGTH = 3
    MAX_USERNAME_LENGTH = 50
    MIN_PASSWORD_LENGTH = 8
    MAX_PASSWORD_LENGTH = 128
    MAX_TITLE_LENGTH = 200
    MAX_NOTES_LENGTH = 2000
    MAX_ICON_NAME_LENGTH = 50
    MAX_TODO_TEXT_LENGTH = 500
    MAX_ACTION_LENGTH = 200
    
    # Regex patterns
    USERNAME_PATTERN = r'^[a-zA-Z0-9_-]+$'
    TIME_PATTERN = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$'
    ICON_NAME_PATTERN = r'^[a-zA-Z0-9_-]+$'
    
    # Allowed HTML tags for sanitization (very restrictive)
    ALLOWED_TAGS = []  # No HTML tags allowed
    ALLOWED_ATTRIBUTES = {}
    
    # Day of week validation
    VALID_DAYS_OF_WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    # Day of month range
    MIN_DAY_OF_MONTH = 1
    MAX_DAY_OF_MONTH = 31


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string input to prevent XSS and other injection attacks
    """
    if not isinstance(value, str):
        raise ValueError("Value must be a string")
    
    # Remove any HTML tags
    sanitized = bleach.clean(
        value,
        tags=ValidationConstants.ALLOWED_TAGS,
        attributes=ValidationConstants.ALLOWED_ATTRIBUTES,
        strip=True
    )
    
    # HTML escape any remaining special characters
    sanitized = html.escape(sanitized)
    
    # Strip whitespace
    sanitized = sanitized.strip()
    
    # Check length if specified
    if max_length and len(sanitized) > max_length:
        raise ValueError(f"String length exceeds maximum of {max_length} characters")
    
    return sanitized


def validate_username(value: str) -> str:
    """Validate username format and length"""
    if not isinstance(value, str):
        raise ValueError("Username must be a string")
    
    value = value.strip().lower()
    
    if len(value) < ValidationConstants.MIN_USERNAME_LENGTH:
        raise ValueError(f"Username must be at least {ValidationConstants.MIN_USERNAME_LENGTH} characters long")
    
    if len(value) > ValidationConstants.MAX_USERNAME_LENGTH:
        raise ValueError(f"Username must not exceed {ValidationConstants.MAX_USERNAME_LENGTH} characters")
    
    if not re.match(ValidationConstants.USERNAME_PATTERN, value):
        raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
    
    return value


def validate_password(value: str) -> str:
    """Validate password strength"""
    if not isinstance(value, str):
        raise ValueError("Password must be a string")
    
    if len(value) < ValidationConstants.MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {ValidationConstants.MIN_PASSWORD_LENGTH} characters long")
    
    if len(value) > ValidationConstants.MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must not exceed {ValidationConstants.MAX_PASSWORD_LENGTH} characters")
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', value):
        raise ValueError("Password must contain at least one uppercase letter")
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', value):
        raise ValueError("Password must contain at least one lowercase letter")
    
    # Check for at least one digit
    if not re.search(r'\d', value):
        raise ValueError("Password must contain at least one digit")
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
        raise ValueError("Password must contain at least one special character")
    
    return value


def validate_time_format(value: str) -> str:
    """Validate time format (HH:MM)"""
    if not isinstance(value, str):
        raise ValueError("Time must be a string")
    
    value = value.strip()
    
    if not re.match(ValidationConstants.TIME_PATTERN, value):
        raise ValueError("Time must be in HH:MM format (24-hour)")
    
    return value


def validate_icon_name(value: str) -> str:
    """Validate icon name format"""
    if not isinstance(value, str):
        raise ValueError("Icon name must be a string")
    
    value = value.strip()
    
    if len(value) > ValidationConstants.MAX_ICON_NAME_LENGTH:
        raise ValueError(f"Icon name must not exceed {ValidationConstants.MAX_ICON_NAME_LENGTH} characters")
    
    if not re.match(ValidationConstants.ICON_NAME_PATTERN, value):
        raise ValueError("Icon name can only contain letters, numbers, underscores, and hyphens")
    
    return value


def validate_days_of_week(value: List[str]) -> List[str]:
    """Validate days of week list"""
    if not isinstance(value, list):
        raise ValueError("Days of week must be a list")
    
    if len(value) == 0:
        return value
    
    # Normalize and validate each day
    normalized_days = []
    for day in value:
        if not isinstance(day, str):
            raise ValueError("Each day must be a string")
        
        day_lower = day.strip().lower()
        if day_lower not in ValidationConstants.VALID_DAYS_OF_WEEK:
            raise ValueError(f"Invalid day of week: {day}. Must be one of: {', '.join(ValidationConstants.VALID_DAYS_OF_WEEK)}")
        
        normalized_days.append(day_lower)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_days = []
    for day in normalized_days:
        if day not in seen:
            seen.add(day)
            unique_days.append(day)
    
    return unique_days


def validate_day_of_month(value: int) -> int:
    """Validate day of month"""
    if not isinstance(value, int):
        raise ValueError("Day of month must be an integer")
    
    if value < ValidationConstants.MIN_DAY_OF_MONTH or value > ValidationConstants.MAX_DAY_OF_MONTH:
        raise ValueError(f"Day of month must be between {ValidationConstants.MIN_DAY_OF_MONTH} and {ValidationConstants.MAX_DAY_OF_MONTH}")
    
    return value


def validate_datetime_not_past(value: datetime, field_name: str = "datetime") -> datetime:
    """Validate that datetime is not in the past (UTC-aware)"""
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime object")

    # Convert naive datetimes to UTC-aware (assume UTC if naive)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    if value < now_utc:
        raise ValueError(f"{field_name} cannot be in the past")
    return value


def validate_end_date_after_start(start_date: datetime, end_date: Optional[datetime]) -> Optional[datetime]:
    """Validate that end date is after start date (UTC-aware)"""
    if end_date is None:
        return end_date
    if not isinstance(end_date, datetime):
        raise ValueError("End date must be a datetime object")
    # Convert naive datetimes to UTC-aware (assume UTC if naive)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    if end_date <= start_date:
        raise ValueError("End date must be after start date")
    return end_date


def validate_user_ids(value: List[int]) -> List[int]:
    """Validate user IDs list"""
    if not isinstance(value, list):
        raise ValueError("User IDs must be a list")
    
    for user_id in value:
        if not isinstance(user_id, int):
            raise ValueError("Each user ID must be an integer")
        
        if user_id <= 0:
            raise ValueError("User IDs must be positive integers")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for user_id in value:
        if user_id not in seen:
            seen.add(user_id)
            unique_ids.append(user_id)
    
    return unique_ids


def validate_positive_integer(value: int, field_name: str = "value") -> int:
    """Validate positive integer"""
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    
    return value
