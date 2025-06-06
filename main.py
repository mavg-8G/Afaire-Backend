from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, ValidationError
from typing import List, Optional
from datetime import datetime, date, time, timezone
from enum import Enum
import re
import html
import bleach
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Table, Text, Boolean, Enum as SqlEnum, UniqueConstraint
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session, backref
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from VerificarToken import create_access_token, verify_token, create_refresh_token, verify_refresh_token
from validation_utils import (
    ValidationConstants, sanitize_string, validate_username, validate_password,
    validate_time_format, validate_icon_name, validate_days_of_week,
    validate_day_of_month, validate_datetime_not_past, validate_end_date_after_start,
    validate_user_ids, validate_positive_integer
)
from error_handlers import (
    SecureErrorResponse, handle_database_error, handle_validation_error,
    handle_generic_exception, sanitize_error_message
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()

# Initialize slowapi Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom exception handlers for comprehensive error handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with secure error messages"""
    return handle_validation_error(exc, request)

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions from custom validation"""
    return handle_validation_error(exc, request)

@app.exception_handler(ValidationError)
async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
    """Handle Pydantic ValidationError exceptions"""
    return handle_validation_error(exc, request)

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    """Handle database errors securely"""
    return handle_database_error(exc, request)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent formatting"""
    error_id = SecureErrorResponse.generate_error_id()
    
    # Log the error for tracking
    SecureErrorResponse.log_detailed_error(
        error_id=error_id,
        error=exc,
        request=request,
        additional_context={"status_code": exc.status_code, "detail": exc.detail}
    )
    
    # Return consistent error format
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP_{exc.status_code}",
            "message": sanitize_error_message(str(exc.detail)),
            "error_id": error_id
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions securely"""
    return handle_generic_exception(exc, request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()
db_engine = create_engine("sqlite:///./todo_app.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get the current user from the JWT token

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    user_id = verify_token(token)
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Enums
class CategoryMode(str, Enum):
    personal = "personal"
    work = "work"
    both = "both"

class RepeatMode(str, Enum):
    none = "none"
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"

# Association tables
activity_user = Table(
    'activity_user', Base.metadata,
    Column('activity_id', ForeignKey('activities.id'), primary_key=True),
    Column('user_id', ForeignKey('users.id'), primary_key=True)
)

# Models
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)

class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    icon_name = Column(String, nullable=False)
    mode = Column(SqlEnum(CategoryMode), nullable=False)

class Activity(Base):
    __tablename__ = 'activities'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    time = Column(String, nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    repeat_mode = Column(SqlEnum(RepeatMode), default=RepeatMode.none, nullable=False)
    end_date = Column(DateTime, nullable=True)
    days_of_week = Column(String, nullable=True)
    day_of_month = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    mode = Column(SqlEnum(CategoryMode), nullable=False)
    category = relationship("Category")
    responsibles = relationship("User", secondary=activity_user, backref="activities")
    todos = relationship("Todo", back_populates="activity", cascade="all, delete-orphan")

class ActivityOccurrence(Base):
    __tablename__ = 'activity_occurrences'
    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activities.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    complete = Column(Boolean, default=False, nullable=False)
    activity = relationship("Activity", backref=backref("occurrences", cascade="all, delete-orphan"))

class Todo(Base):
    __tablename__ = 'todos'
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    complete = Column(Boolean, default=False, nullable=False)
    activity_id = Column(Integer, ForeignKey("activities.id"), nullable=False)
    activity = relationship("Activity", back_populates="todos")

class History(Base):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User")

class RefreshToken(Base):
    __tablename__ = 'refresh_tokens'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    token = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked = Column(Boolean, default=False)
    user = relationship('User')
    __table_args__ = (UniqueConstraint('user_id', 'token', name='_user_token_uc'),)

Base.metadata.create_all(bind=db_engine)
# Seed default admin user for tests
from sqlalchemy.orm import Session
_db = SessionLocal()
try:
    if not _db.query(User).filter(User.username == 'testuser').first():
        _db.add(User(name='Test User', username='testuser', hashed_password=get_password_hash('TestPass123!'), is_admin=True))
        _db.commit()
except Exception:
    _db.rollback()
finally:
    _db.close()

# Schemas with comprehensive validation
class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1, max_length=ValidationConstants.MAX_PASSWORD_LENGTH)
    new_password: str = Field(..., min_length=ValidationConstants.MIN_PASSWORD_LENGTH, max_length=ValidationConstants.MAX_PASSWORD_LENGTH)
    
    @field_validator('old_password', 'new_password')
    @classmethod
    def sanitize_passwords(cls, v):
        if not isinstance(v, str):
            raise ValueError("Password must be a string")
        return v.strip()
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password_strength(cls, v):
        return validate_password(v)


class UserCreate(BaseModel):
    name: str = Field(..., min_length=ValidationConstants.MIN_NAME_LENGTH, max_length=ValidationConstants.MAX_NAME_LENGTH)
    username: str = Field(..., min_length=ValidationConstants.MIN_USERNAME_LENGTH, max_length=ValidationConstants.MAX_USERNAME_LENGTH)
    password: str = Field(..., min_length=ValidationConstants.MIN_PASSWORD_LENGTH, max_length=ValidationConstants.MAX_PASSWORD_LENGTH)
    is_admin: bool = Field(default=False)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        return sanitize_string(v, ValidationConstants.MAX_NAME_LENGTH)
    
    @field_validator('username')
    @classmethod
    def validate_username_format(cls, v):
        return validate_username(v)
    
    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v):
        return validate_password(v)
    
    @field_validator('is_admin')
    @classmethod
    def validate_is_admin(cls, v):
        if not isinstance(v, bool):
            raise ValueError("is_admin must be a boolean")
        return v


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=ValidationConstants.MIN_NAME_LENGTH, max_length=ValidationConstants.MAX_NAME_LENGTH)
    username: Optional[str] = Field(None, min_length=ValidationConstants.MIN_USERNAME_LENGTH, max_length=ValidationConstants.MAX_USERNAME_LENGTH)
    password: Optional[str] = Field(None, min_length=ValidationConstants.MIN_PASSWORD_LENGTH, max_length=ValidationConstants.MAX_PASSWORD_LENGTH)
    is_admin: Optional[bool] = None
    
    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            return sanitize_string(v, ValidationConstants.MAX_NAME_LENGTH)
        return v
    
    @field_validator('username', mode='before')
    @classmethod
    def validate_username_format(cls, v):
        if v is not None:
            return validate_username(v)
        return v
    
    @field_validator('password', mode='before')
    @classmethod
    def validate_password_strength(cls, v):
        if v is not None:
            return validate_password(v)
        return v
    
    @field_validator('is_admin')
    @classmethod
    def validate_is_admin(cls, v):
        if v is not None and not isinstance(v, bool):
            raise ValueError("is_admin must be a boolean")
        return v


class CategoryCreate(BaseModel):
    name: str = Field(..., min_length=ValidationConstants.MIN_NAME_LENGTH, max_length=ValidationConstants.MAX_NAME_LENGTH)
    icon_name: str = Field(..., min_length=1, max_length=ValidationConstants.MAX_ICON_NAME_LENGTH)
    mode: CategoryMode
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        return sanitize_string(v, ValidationConstants.MAX_NAME_LENGTH)
    
    @field_validator('icon_name')
    @classmethod
    def validate_icon_name_format(cls, v):
        return validate_icon_name(v)
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if not isinstance(v, CategoryMode):
            raise ValueError(f"Mode must be one of: {', '.join([mode.value for mode in CategoryMode])}")
        return v


class CategoryUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=ValidationConstants.MIN_NAME_LENGTH, max_length=ValidationConstants.MAX_NAME_LENGTH)
    icon_name: Optional[str] = Field(None, min_length=1, max_length=ValidationConstants.MAX_ICON_NAME_LENGTH)
    mode: Optional[CategoryMode] = None
    
    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            return sanitize_string(v, ValidationConstants.MAX_NAME_LENGTH)
        return v
    
    @field_validator('icon_name', mode='before')
    @classmethod
    def validate_icon_name_format(cls, v):
        if v is not None:
            return validate_icon_name(v)
        return v
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if v is not None and not isinstance(v, CategoryMode):
            raise ValueError(f"Mode must be one of: {', '.join([mode.value for mode in CategoryMode])}")
        return v


class TodoCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=ValidationConstants.MAX_TODO_TEXT_LENGTH)
    complete: bool = Field(default=False)
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        return sanitize_string(v, ValidationConstants.MAX_TODO_TEXT_LENGTH)
    
    @field_validator('complete')
    @classmethod
    def validate_complete(cls, v):
        if not isinstance(v, bool):
            raise ValueError("complete must be a boolean")
        return v


class TodoUpdate(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=ValidationConstants.MAX_TODO_TEXT_LENGTH)
    complete: Optional[bool] = None
    
    @field_validator('text', mode='before')
    @classmethod
    def validate_text(cls, v):
        if v is not None:
            return sanitize_string(v, ValidationConstants.MAX_TODO_TEXT_LENGTH)
        return v
    
    @field_validator('complete')
    @classmethod
    def validate_complete(cls, v):
        if v is not None and not isinstance(v, bool):
            raise ValueError("complete must be a boolean")
        return v


class ActivityResponse(BaseModel):
    id: int
    title: str
    start_date: datetime
    time: str
    category_id: int
    repeat_mode: RepeatMode
    end_date: Optional[datetime]
    days_of_week: Optional[str]
    day_of_month: Optional[int]
    notes: Optional[str]
    mode: CategoryMode
    responsible_ids: List[int]

    # Pydantic V2: enable model to read from ORM objects
    model_config = ConfigDict(from_attributes=True)


class ActivityCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=ValidationConstants.MAX_TITLE_LENGTH)
    start_date: datetime = Field(...)
    time: str = Field(..., min_length=1, max_length=5)
    category_id: int = Field(..., gt=0)
    repeat_mode: RepeatMode = Field(default=RepeatMode.none)
    end_date: Optional[datetime] = None
    days_of_week: Optional[List[str]] = None
    day_of_month: Optional[int] = Field(None, ge=ValidationConstants.MIN_DAY_OF_MONTH, le=ValidationConstants.MAX_DAY_OF_MONTH)
    notes: Optional[str] = Field(None, max_length=ValidationConstants.MAX_NOTES_LENGTH)
    mode: CategoryMode
    responsible_ids: List[int] = Field(default=[])
    todos: List[TodoCreate] = Field(default=[])
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        return sanitize_string(v, ValidationConstants.MAX_TITLE_LENGTH)
    
    @field_validator('time')
    @classmethod
    def validate_time_format(cls, v):
        return validate_time_format(v)
    
    @field_validator('category_id')
    @classmethod
    def validate_category_id(cls, v):
        return validate_positive_integer(v, "category_id")
    
    @field_validator('repeat_mode')
    @classmethod
    def validate_repeat_mode(cls, v):
        if not isinstance(v, RepeatMode):
            raise ValueError(f"Repeat mode must be one of: {', '.join([mode.value for mode in RepeatMode])}")
        return v
    
    @field_validator('days_of_week', mode='before')
    @classmethod
    def validate_days_of_week_list(cls, v):
        if v is not None:
            return validate_days_of_week(v)
        return v
    
    @field_validator('day_of_month', mode='before')
    @classmethod
    def validate_day_of_month_range(cls, v):
        if v is not None:
            return validate_day_of_month(v)
        return v
    
    @field_validator('notes', mode='before')
    @classmethod
    def validate_notes(cls, v):
        if v is not None:
            return sanitize_string(v, ValidationConstants.MAX_NOTES_LENGTH)
        return v
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if not isinstance(v, CategoryMode):
            raise ValueError(f"Mode must be one of: {', '.join([mode.value for mode in CategoryMode])}")
        return v
    
    @field_validator('responsible_ids')
    @classmethod
    def validate_responsible_ids(cls, v):
        return validate_user_ids(v)
    
    @field_validator('todos')
    @classmethod
    def validate_todos_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("todos must be a list")
        if len(v) > 50:  # Reasonable limit
            raise ValueError("Too many todos (maximum 50)")
        return v
    
    @model_validator(mode='after')
    def validate_end_date_logic(self):
        start_date = self.start_date
        end_date = self.end_date
        repeat_mode = self.repeat_mode
        
        if start_date and end_date:
            self.end_date = validate_end_date_after_start(start_date, end_date)
        
        # If repeat mode is not 'none', validate related fields
        if repeat_mode and repeat_mode != RepeatMode.none:
            if repeat_mode == RepeatMode.weekly and not self.days_of_week:
                raise ValueError("days_of_week is required for weekly repeat mode")
            elif repeat_mode == RepeatMode.monthly and not self.day_of_month:
                raise ValueError("day_of_month is required for monthly repeat mode")
        
        return self


class ActivityUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=ValidationConstants.MAX_TITLE_LENGTH)
    start_date: Optional[datetime] = None
    time: Optional[str] = Field(None, min_length=1, max_length=5)
    category_id: Optional[int] = Field(None, gt=0)
    repeat_mode: Optional[RepeatMode] = None
    end_date: Optional[datetime] = None
    days_of_week: Optional[List[str]] = None
    day_of_month: Optional[int] = Field(None, ge=ValidationConstants.MIN_DAY_OF_MONTH, le=ValidationConstants.MAX_DAY_OF_MONTH)
    notes: Optional[str] = Field(None, max_length=ValidationConstants.MAX_NOTES_LENGTH)
    mode: Optional[CategoryMode] = None
    responsible_ids: Optional[List[int]] = None
    
    @field_validator('title', mode='before')
    @classmethod
    def validate_title(cls, v):
        if v is not None:
            return sanitize_string(v, ValidationConstants.MAX_TITLE_LENGTH)
        return v
    
    @field_validator('time', mode='before')
    @classmethod
    def validate_time_format(cls, v):
        if v is not None:
            return validate_time_format(v)
        return v
    
    @field_validator('category_id', mode='before')
    @classmethod
    def validate_category_id(cls, v):
        if v is not None:
            return validate_positive_integer(v, "category_id")
        return v
    
    @field_validator('repeat_mode')
    @classmethod
    def validate_repeat_mode(cls, v):
        if v is not None and not isinstance(v, RepeatMode):
            raise ValueError(f"Repeat mode must be one of: {', '.join([mode.value for mode in RepeatMode])}")
        return v
    
    @field_validator('days_of_week', mode='before')
    @classmethod
    def validate_days_of_week_list(cls, v):
        if v is not None:
            return validate_days_of_week(v)
        return v
    
    @field_validator('day_of_month', mode='before')
    @classmethod
    def validate_day_of_month_range(cls, v):
        if v is not None:
            return validate_day_of_month(v)
        return v
    
    @field_validator('notes', mode='before')
    @classmethod
    def validate_notes(cls, v):
        if v is not None:
            return sanitize_string(v, ValidationConstants.MAX_NOTES_LENGTH)
        return v
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        if v is not None and not isinstance(v, CategoryMode):
            raise ValueError(f"Mode must be one of: {', '.join([mode.value for mode in CategoryMode])}")
        return v
    
    @field_validator('responsible_ids', mode='before')
    @classmethod
    def validate_responsible_ids(cls, v):
        if v is not None:
            return validate_user_ids(v)
        return v


class ActivityOccurrenceResponse(BaseModel):
    id: int
    activity_id: int
    date: datetime
    complete: bool
    activity_title: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ActivityOccurrenceCreate(BaseModel):
    activity_id: int = Field(..., gt=0)
    date: datetime = Field(...)
    complete: bool = Field(default=False)
    
    @field_validator('activity_id')
    @classmethod
    def validate_activity_id(cls, v):
        return validate_positive_integer(v, "activity_id")
    
    @field_validator('complete')
    @classmethod
    def validate_complete(cls, v):
        if not isinstance(v, bool):
            raise ValueError("complete must be a boolean")
        return v


class ActivityOccurrenceUpdate(BaseModel):
    date: Optional[datetime] = None
    complete: Optional[bool] = None
    
    @field_validator('complete')
    @classmethod
    def validate_complete(cls, v):
        if v is not None and not isinstance(v, bool):
            raise ValueError("complete must be a boolean")
        return v


class HistoryCreate(BaseModel):
    action: str = Field(..., min_length=1, max_length=ValidationConstants.MAX_ACTION_LENGTH)
    user_id: int = Field(..., gt=0)
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        return sanitize_string(v, ValidationConstants.MAX_ACTION_LENGTH)
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        return validate_positive_integer(v, "user_id")
# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utils
def get_password_hash(password):
    return pwd_context.hash(password)

def record_history(db: Session, user_id: int, action: str):
    db.add(History(user_id=user_id, action=action))
    db.commit()

# Routes
@app.post("/token")
@limiter.limit("5/minute")  # Limit to 5 login attempts per minute per IP
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        # Sanitize input
        username = sanitize_string(form_data.username, ValidationConstants.MAX_USERNAME_LENGTH).lower()
        password = form_data.password
        
        # Additional validation
        if not username:
            return SecureErrorResponse.validation_error("Invalid credentials")
        if not password:
            return SecureErrorResponse.validation_error("Invalid credentials")
        
        # Validate username format
        try:
            validate_username(username)
        except ValueError:
            # Don't reveal that username format is invalid for security
            return SecureErrorResponse.authentication_error("Invalid credentials")
        
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return SecureErrorResponse.authentication_error("Invalid credentials")
        
        password_valid = pwd_context.verify(password, user.hashed_password)
        
        if not password_valid:
            return SecureErrorResponse.authentication_error("Invalid credentials")
        try:
            access_token = create_access_token(data={"sub": str(user.id)})
            refresh_token = create_refresh_token(data={"sub": str(user.id)})
            # Store refresh token in DB
            db.add(RefreshToken(user_id=user.id, token=refresh_token))
            db.commit()
            response = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "user_id": user.id,
                "username": user.username,
                "is_admin": user.is_admin
            }
            return response
        except Exception as token_error:
            error_id = SecureErrorResponse.generate_error_id()
            SecureErrorResponse.log_detailed_error(
                error_id=error_id,
                error=token_error,
                additional_context={"operation": "token_creation", "user_id": user.id}
            )
            return SecureErrorResponse.internal_server_error("Authentication service temporarily unavailable")
    except Exception as e:
        # Log detailed error and return generic response
        error_id = SecureErrorResponse.generate_error_id()
        SecureErrorResponse.log_detailed_error(
            error_id=error_id,
            error=e,
            additional_context={"operation": "login"}
        )
        return SecureErrorResponse.internal_server_error("Authentication service temporarily unavailable")


@app.post("/refresh-token")
def refresh_token_endpoint(refresh_token: str, db: Session = Depends(get_db)):
    """Exchange a valid refresh token for a new access token."""
    try:
        user_id = verify_refresh_token(refresh_token)
        token_obj = db.query(RefreshToken).filter_by(token=refresh_token, revoked=False).first()
        if not token_obj or str(token_obj.user_id) != str(user_id):
            return SecureErrorResponse.authentication_error("Invalid or expired refresh token")
        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            return SecureErrorResponse.authentication_error("User not found")
        new_access_token = create_access_token(data={"sub": str(user.id)})
        return {"access_token": new_access_token, "token_type": "bearer"}
    except Exception as e:
        return SecureErrorResponse.authentication_error("Invalid or expired refresh token")


@app.post("/logout")
def logout(refresh_token: str, db: Session = Depends(get_db)):
    """Invalidate a refresh token (logout)."""
    try:
        user_id = verify_refresh_token(refresh_token)
        token_obj = db.query(RefreshToken).filter_by(token=refresh_token, revoked=False).first()
        if not token_obj or str(token_obj.user_id) != str(user_id):
            return SecureErrorResponse.authentication_error("Invalid refresh token")
        token_obj.revoked = True
        db.commit()
        return {"detail": "Logged out successfully"}
    except Exception as e:
        db.rollback()
        return SecureErrorResponse.authentication_error("Invalid or expired refresh token")


@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    try:
        # Validate user_id
        if user_id <= 0:
            return SecureErrorResponse.validation_error("Invalid user ID")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return SecureErrorResponse.not_found_error("User not found")
        return user
    
    except Exception as e:
        return handle_generic_exception(e)

@app.post("/users")
@limiter.limit("3/minute")  # Limit to 3 user creations per minute per IP
def create_user(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    try:
        # Additional server-side validation
        if not user.name.strip():
            return SecureErrorResponse.validation_error("Name cannot be empty")
        
        # Check if username already exists (case-insensitive)
        existing_user = db.query(User).filter(User.username.ilike(user.username)).first()
        if existing_user:
            return SecureErrorResponse.conflict_error("Username already exists")
        
        # Additional password validation in case frontend validation was bypassed
        try:
            validate_password(user.password)
        except ValueError as e:
            return SecureErrorResponse.validation_error(str(e))
        
        db_user = User(
            name=user.name,
            username=user.username.lower(),  # Store username in lowercase
            hashed_password=get_password_hash(user.password),
            is_admin=user.is_admin
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    
    except Exception as e:
        # Rollback transaction on error
        db.rollback()
        return handle_database_error(e)


@app.put("/users/{user_id}")
def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Validate user_id
        if user_id <= 0:
            return SecureErrorResponse.validation_error("Invalid user ID")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return SecureErrorResponse.not_found_error("User not found")

        # Authorization: Only the user themselves or admin can update
        if current_user.id != user_id and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to update this user")
        
        # Check if username already exists (case-insensitive, excluding current user)
        if user_update.username is not None:
            existing_user = db.query(User).filter(
                User.username.ilike(user_update.username), 
                User.id != user_id
            ).first()
            if existing_user:
                return SecureErrorResponse.conflict_error("Username already exists")
        
        # Additional password validation if password is being updated
        if user_update.password is not None:
            try:
                validate_password(user_update.password)
            except ValueError as e:
                return SecureErrorResponse.validation_error(str(e))
        
        # Update fields
        if user_update.name is not None:
            if not user_update.name.strip():
                return SecureErrorResponse.validation_error("Name cannot be empty")
            user.name = user_update.name
        if user_update.username is not None:
            user.username = user_update.username.lower()  # Store username in lowercase
        if user_update.password is not None:
            user.hashed_password = get_password_hash(user_update.password)
        if user_update.is_admin is not None:
            user.is_admin = user_update.is_admin
        
        db.commit()
        db.refresh(user)
        return user
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.post("/users/{user_id}/change-password")
@limiter.limit("3/minute")  # Limit to 3 password changes per minute per IP
def change_password(request: Request, user_id: int, req: ChangePasswordRequest, db: Session = Depends(get_db)):
    try:
        # Validate user_id
        if user_id <= 0:
            return SecureErrorResponse.validation_error("Invalid user ID")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return SecureErrorResponse.not_found_error("User not found")
        
        # Verify old password
        if not pwd_context.verify(req.old_password, user.hashed_password):
            return SecureErrorResponse.authentication_error("Old password is incorrect")
        
        # Ensure new password is different from old password
        if pwd_context.verify(req.new_password, user.hashed_password):
            return SecureErrorResponse.validation_error("New password must be different from current password")
        
        # Additional validation for new password (already validated in Pydantic model)
        try:
            validate_password(req.new_password)
        except ValueError as e:
            return SecureErrorResponse.validation_error(str(e))
        
        user.hashed_password = get_password_hash(req.new_password)
        # Invalidate all refresh tokens for this user
        db.query(RefreshToken).filter_by(user_id=user_id, revoked=False).update({"revoked": True})
        db.commit()
        return {"detail": "Password updated successfully"}
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)


@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Validate user_id
        if user_id <= 0:
            return SecureErrorResponse.validation_error("Invalid user ID")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return SecureErrorResponse.not_found_error("User not found")

        # Authorization: Only the user themselves or admin can delete
        if current_user.id != user_id and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to delete this user")
        
        # Check if user has associated activities before deletion
        user_activities = db.query(Activity).join(activity_user).filter(activity_user.c.user_id == user_id).first()
        if user_activities:
            return SecureErrorResponse.conflict_error("Cannot delete user with associated activities")
        
        db.delete(user)
        db.commit()
        return {"detail": "User deleted successfully"}
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.post("/categories")
def create_category(category: CategoryCreate, db: Session = Depends(get_db)):
    try:
        # Additional server-side validation
        if not category.name.strip():
            return SecureErrorResponse.validation_error("Category name cannot be empty")
        
        if not category.icon_name.strip():
            return SecureErrorResponse.validation_error("Icon name cannot be empty")
        
        # Check for duplicate category names (case-insensitive)
        existing_category = db.query(Category).filter(Category.name.ilike(category.name)).first()
        if existing_category:
            return SecureErrorResponse.conflict_error("Category name already exists")
        
        db_category = Category(**category.model_dump())
        db.add(db_category)
        db.commit()
        db.refresh(db_category)
        return db_category
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.get("/categories")
def get_categories(db: Session = Depends(get_db)):
    return db.query(Category).all()

@app.get("/categories/{category_id}")
def get_category(category_id: int, db: Session = Depends(get_db)):
    try:
        # Validate category_id
        if category_id <= 0:
            return SecureErrorResponse.validation_error("Invalid category ID")
        
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            return SecureErrorResponse.not_found_error("Category not found")
        return category
    
    except Exception as e:
        return handle_generic_exception(e)

@app.put("/categories/{category_id}")
def update_category(category_id: int, category_update: CategoryUpdate, db: Session = Depends(get_db)):
    try:
        # Validate category_id
        if category_id <= 0:
            return SecureErrorResponse.validation_error("Invalid category ID")
        
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            return SecureErrorResponse.not_found_error("Category not found")
        
        # Check for duplicate category names if name is being updated
        if category_update.name is not None:
            if not category_update.name.strip():
                return SecureErrorResponse.validation_error("Category name cannot be empty")
            
            existing_category = db.query(Category).filter(
                Category.name.ilike(category_update.name), 
                Category.id != category_id
            ).first()
            if existing_category:
                return SecureErrorResponse.conflict_error("Category name already exists")
        
        if category_update.icon_name is not None and not category_update.icon_name.strip():
            return SecureErrorResponse.validation_error("Icon name cannot be empty")
        
        update_data = category_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(category, key, value)
        db.commit()
        db.refresh(category)
        return category
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.delete("/categories/{category_id}")
def delete_category(category_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Only admins can delete categories
        if not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to delete categories")
        
        # Validate category_id
        if category_id <= 0:
            return SecureErrorResponse.validation_error("Invalid category ID")
        
        category = db.query(Category).filter(Category.id == category_id).first()
        if not category:
            return SecureErrorResponse.not_found_error("Category not found")
        
        # Check if category is being used by any activities
        category_activities = db.query(Activity).filter(Activity.category_id == category_id).first()
        if category_activities:
            return SecureErrorResponse.conflict_error("Cannot delete category that is being used by activities")
        
        db.delete(category)
        db.commit()
        return {"detail": "Category deleted successfully"}
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.get("/activities", response_model=List[ActivityResponse])
def get_activities(db: Session = Depends(get_db)):
    activities = db.query(Activity).all()
    # Map responsibles to responsible_ids
    return [
        ActivityResponse(
            id=a.id,
            title=a.title,
            start_date=a.start_date,
            time=a.time,
            category_id=a.category_id,
            repeat_mode=a.repeat_mode,
            end_date=a.end_date,
            days_of_week=a.days_of_week,
            day_of_month=a.day_of_month,
            notes=a.notes,
            mode=a.mode,
            responsible_ids=[u.id for u in a.responsibles]
        )
        for a in activities
    ]

@app.post("/activities")
def create_activity(activity: ActivityCreate, db: Session = Depends(get_db)):
    try:
        # Additional server-side validation
        if not activity.title.strip():
            return SecureErrorResponse.validation_error("Activity title cannot be empty")
        
        # Validate start_date is not too far in the past (allow some flexibility for scheduling)
        if activity.start_date.tzinfo is None:
            activity_start_utc = activity.start_date.replace(tzinfo=timezone.utc)
        else:
            activity_start_utc = activity.start_date.astimezone(timezone.utc)
        if activity_start_utc < datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0):
            return SecureErrorResponse.validation_error("Start date cannot be in the past")
        
        # Verify category exists and is valid
        category = db.query(Category).filter(Category.id == activity.category_id).first()
        if not category:
            return SecureErrorResponse.not_found_error("Category not found")
        
        # Validate mode matches category mode or is compatible
        if activity.mode not in [category.mode, CategoryMode.both]:
            if category.mode != CategoryMode.both:
                return SecureErrorResponse.validation_error("Activity mode must be compatible with category mode")
        
        # Verify all responsible users exist
        existing_users = []
        if activity.responsible_ids:
            unique_ids = list(set(activity.responsible_ids))
            if len(unique_ids) != len(activity.responsible_ids):
                return SecureErrorResponse.validation_error("Duplicate user IDs found in responsible_ids")
            existing_users = db.query(User).filter(User.id.in_(unique_ids)).all()
            if len(existing_users) != len(unique_ids):
                missing_ids = set(unique_ids) - {user.id for user in existing_users}
                return SecureErrorResponse.not_found_error(f"Users not found: {list(missing_ids)}")
        
        # Validate repeat mode logic
        if activity.repeat_mode == RepeatMode.weekly:
            if not activity.days_of_week:
                return SecureErrorResponse.validation_error("days_of_week is required for weekly repeat mode")
            if len(activity.days_of_week) == 0:
                return SecureErrorResponse.validation_error("At least one day must be specified for weekly repeat")
        elif activity.repeat_mode == RepeatMode.monthly:
            if activity.day_of_month is None:
                return SecureErrorResponse.validation_error("day_of_month is required for monthly repeat mode")
        
        # Validate end_date logic
        if activity.end_date:
            # Always compare in UTC
            if activity.end_date.tzinfo is None:
                end_date_utc = activity.end_date.replace(tzinfo=timezone.utc)
            else:
                end_date_utc = activity.end_date.astimezone(timezone.utc)
            if end_date_utc <= activity_start_utc:
                return SecureErrorResponse.validation_error("End date must be after start date")
            if activity.repeat_mode == RepeatMode.none:
                return SecureErrorResponse.validation_error("End date is only valid for repeating activities")
        
        # Validate todos
        if len(activity.todos) > 50:
            return SecureErrorResponse.validation_error("Too many todos (maximum 50 per activity)")
        
        # Create the activity
        db_activity = Activity(
            title=activity.title.strip(),
            start_date=activity.start_date,
            time=activity.time,
            category_id=activity.category_id,
            repeat_mode=activity.repeat_mode,
            end_date=activity.end_date,
            days_of_week=','.join(activity.days_of_week or []),
            day_of_month=activity.day_of_month,
            notes=activity.notes.strip() if activity.notes else None,
            mode=activity.mode
        )
        
        # Assign responsible users
        if existing_users:
            db_activity.responsibles = existing_users
        
        # Save activity first
        db.add(db_activity)
        db.commit()
        db.refresh(db_activity)
        
        # Add TODOs after activity has an ID
        for todo_data in activity.todos:
            if not todo_data.text.strip():
                return SecureErrorResponse.validation_error("Todo text cannot be empty")
            
            db_todo = Todo(
                text=todo_data.text.strip(), 
                complete=todo_data.complete,
                activity_id=db_activity.id
            )
            db.add(db_todo)
        
        # Final commit for TODOs
        db.commit()
        db.refresh(db_activity)
        
        print(f"Activity created successfully with ID: {db_activity.id}")  # Debug log
        return db_activity
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.get("/activities/{activity_id}", response_model=ActivityResponse)
def get_activity(activity_id: int, db: Session = Depends(get_db)):
    try:
        # Validate activity_id
        if activity_id <= 0:
            return SecureErrorResponse.validation_error("Invalid activity ID")
        
        activity = db.query(Activity).filter(Activity.id == activity_id).first()
        if not activity:
            return SecureErrorResponse.not_found_error("Activity not found")
        
        return ActivityResponse(
            id=activity.id,
            title=activity.title,
            start_date=activity.start_date,
            time=activity.time,
            category_id=activity.category_id,
            repeat_mode=activity.repeat_mode,
            end_date=activity.end_date,
            days_of_week=activity.days_of_week,
            day_of_month=activity.day_of_month,
            notes=activity.notes,
            mode=activity.mode,
            responsible_ids=[u.id for u in activity.responsibles]
        )
    
    except Exception as e:
        return handle_generic_exception(e)

@app.put("/activities/{activity_id}")
def update_activity(activity_id: int, activity_update: ActivityUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Validate activity_id
        if activity_id <= 0:
            return SecureErrorResponse.validation_error("Invalid activity ID")
        activity = db.query(Activity).filter(Activity.id == activity_id).first()
        if not activity:
            return SecureErrorResponse.not_found_error("Activity not found")
        # Authorization: Only responsible users or admins can update
        if current_user not in activity.responsibles and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to update this activity")
        update_data = activity_update.model_dump(exclude_unset=True)
        # Additional validation for updated fields
        if "title" in update_data and update_data["title"]:
            if not update_data["title"].strip():
                return SecureErrorResponse.validation_error("Activity title cannot be empty")
            update_data["title"] = update_data["title"].strip()
        # Validate category_id if being updated
        if "category_id" in update_data:
            category = db.query(Category).filter(Category.id == update_data["category_id"]).first()
            if not category:
                return SecureErrorResponse.not_found_error("Category not found")
            new_mode = update_data.get("mode", activity.mode)
            if new_mode not in [category.mode, CategoryMode.both]:
                if category.mode != CategoryMode.both:
                    return SecureErrorResponse.validation_error("Activity mode must be compatible with category mode")
        # Validate start_date if being updated
        if "start_date" in update_data:
            start_date = update_data["start_date"]
            if start_date.tzinfo is None:
                start_date_utc = start_date.replace(tzinfo=timezone.utc)
            else:
                start_date_utc = start_date.astimezone(timezone.utc)
            if start_date_utc < datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0):
                return SecureErrorResponse.validation_error("Start date cannot be in the past")
        else:
            start_date_utc = activity.start_date
            if start_date_utc.tzinfo is None:
                start_date_utc = start_date_utc.replace(tzinfo=timezone.utc)
            else:
                start_date_utc = start_date_utc.astimezone(timezone.utc)
        # Validate end_date logic if being updated
        if "end_date" in update_data and update_data["end_date"]:
            end_date = update_data["end_date"]
            if end_date.tzinfo is None:
                end_date_utc = end_date.replace(tzinfo=timezone.utc)
            else:
                end_date_utc = end_date.astimezone(timezone.utc)
            if end_date_utc <= start_date_utc:
                return SecureErrorResponse.validation_error("End date must be after start date")
        # Validate repeat mode logic
        if "repeat_mode" in update_data:
            repeat_mode = update_data["repeat_mode"]
            if repeat_mode == RepeatMode.weekly:
                days_of_week = update_data.get("days_of_week", activity.days_of_week)
                if not days_of_week or (isinstance(days_of_week, str) and not days_of_week.strip()):
                    return SecureErrorResponse.validation_error("days_of_week is required for weekly repeat mode")
            elif repeat_mode == RepeatMode.monthly:
                day_of_month = update_data.get("day_of_month", activity.day_of_month)
                if day_of_month is None:
                    return SecureErrorResponse.validation_error("day_of_month is required for monthly repeat mode")
        # Handle days_of_week conversion
        if "days_of_week" in update_data and update_data["days_of_week"] is not None:
            if isinstance(update_data["days_of_week"], list):
                update_data["days_of_week"] = ",".join(update_data["days_of_week"])
        # Sanitize notes if being updated
        if "notes" in update_data and update_data["notes"]:
            update_data["notes"] = update_data["notes"].strip()
        # Handle responsible_ids separately
        if "responsible_ids" in update_data:
            responsible_ids = update_data.pop("responsible_ids")
            if responsible_ids:
                unique_ids = list(set(responsible_ids))
                if len(unique_ids) != len(responsible_ids):
                    return SecureErrorResponse.validation_error("Duplicate user IDs found in responsible_ids")
                existing_users = db.query(User).filter(User.id.in_(unique_ids)).all()
                if len(existing_users) != len(unique_ids):
                    missing_ids = set(unique_ids) - {user.id for user in existing_users}
                    return SecureErrorResponse.not_found_error(f"Users not found: {list(missing_ids)}")
                activity.responsibles = existing_users
            else:
                activity.responsibles = []
        # Apply other updates
        for key, value in update_data.items():
            setattr(activity, key, value)
        db.commit()
        db.refresh(activity)
        return activity
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.post("/activity-occurrences")
def create_activity_occurrence(occurrence: ActivityOccurrenceCreate, db: Session = Depends(get_db)):
    """Crear una nueva ocurrencia de actividad"""
    try:
        # Verify activity exists
        activity = db.query(Activity).filter(Activity.id == occurrence.activity_id).first()
        if not activity:
            return SecureErrorResponse.not_found_error("Activity not found")
        # Additional validation for occurrence date
        occ_date = occurrence.date
        if occ_date.tzinfo is None:
            occ_date_utc = occ_date.replace(tzinfo=timezone.utc)
        else:
            occ_date_utc = occ_date.astimezone(timezone.utc)
        if occ_date_utc < datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0):
            return SecureErrorResponse.validation_error("Occurrence date cannot be in the past")
        # Check for duplicate occurrence (same activity and date)
        existing_occurrence = db.query(ActivityOccurrence).filter(
            ActivityOccurrence.activity_id == occurrence.activity_id,
            ActivityOccurrence.date == occurrence.date
        ).first()
        if existing_occurrence:
            return SecureErrorResponse.conflict_error("An occurrence for this activity and date already exists")
        db_occurrence = ActivityOccurrence(
            activity_id=occurrence.activity_id,
            date=occurrence.date,
            complete=occurrence.complete
        )
        db.add(db_occurrence)
        db.commit()
        db.refresh(db_occurrence)
        return db_occurrence
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

# Endpoints para Activity Occurrences
@app.get("/activity-occurrences")
def get_all_activity_occurrences(db: Session = Depends(get_db)):
    """Obtener todas las ocurrencias de actividades"""
    occurrences = db.query(ActivityOccurrence).all()
    return [
        ActivityOccurrenceResponse(
            id=occ.id,
            activity_id=occ.activity_id,
            date=occ.date,
            complete=occ.complete,
            activity_title=occ.activity.title if occ.activity else None
        )
        for occ in occurrences
    ]

@app.get("/activity-occurrences/{occurrence_id}")
def get_activity_occurrence(occurrence_id: int, db: Session = Depends(get_db)):
    """Obtener una ocurrencia específica por ID"""
    try:
        # Validate occurrence_id
        if occurrence_id <= 0:
            return SecureErrorResponse.validation_error("Invalid occurrence ID")
        
        occurrence = db.query(ActivityOccurrence).filter(ActivityOccurrence.id == occurrence_id).first()
        if not occurrence:
            return SecureErrorResponse.not_found_error("Activity occurrence not found")
        
        return ActivityOccurrenceResponse(
            id=occurrence.id,
            activity_id=occurrence.activity_id,
            date=occurrence.date,
            complete=occurrence.complete,
            activity_title=occurrence.activity.title if occurrence.activity else None
        )
    
    except Exception as e:
        return handle_generic_exception(e)

@app.put("/activity-occurrences/{occurrence_id}")
def update_activity_occurrence(occurrence_id: int, occurrence_update: ActivityOccurrenceUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Validate occurrence_id
        if occurrence_id <= 0:
            return SecureErrorResponse.validation_error("Invalid occurrence ID")
        
        occurrence = db.query(ActivityOccurrence).filter(ActivityOccurrence.id == occurrence_id).first()
        if not occurrence:
            return SecureErrorResponse.not_found_error("Activity occurrence not found")

        # Authorization: Only responsible users or admins can update
        if current_user not in occurrence.activity.responsibles and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to update this occurrence")

        update_data = occurrence_update.model_dump(exclude_unset=True)
        
        # Additional validation for date if being updated
        if "date" in update_data:
            new_date = update_data["date"]
            if new_date < datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
                return SecureErrorResponse.validation_error("Occurrence date cannot be in the past")
            
            # Check for duplicate occurrence with new date
            if new_date != occurrence.date:
                existing_occurrence = db.query(ActivityOccurrence).filter(
                    ActivityOccurrence.activity_id == occurrence.activity_id,
                    ActivityOccurrence.date == new_date,
                    ActivityOccurrence.id != occurrence_id
                ).first()
                if existing_occurrence:
                    return SecureErrorResponse.conflict_error("An occurrence for this activity and date already exists")
        
        for key, value in update_data.items():
            setattr(occurrence, key, value)
        
        db.commit()
        db.refresh(occurrence)
        
        return ActivityOccurrenceResponse(
            id=occurrence.id,
            activity_id=occurrence.activity_id,
            date=occurrence.date,
            complete=occurrence.complete,
            activity_title=occurrence.activity.title if occurrence.activity else None
        )
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.delete("/activity-occurrences/{occurrence_id}")
def delete_activity_occurrence(occurrence_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Eliminar una ocurrencia de actividad"""
    try:
        # Validate occurrence_id
        if occurrence_id <= 0:
            return SecureErrorResponse.validation_error("Invalid occurrence ID")
        
        occurrence = db.query(ActivityOccurrence).filter(ActivityOccurrence.id == occurrence_id).first()
        if not occurrence:
            return SecureErrorResponse.not_found_error("Activity occurrence not found")

        # Authorization: Only responsible users or admins can delete
        if current_user not in occurrence.activity.responsibles and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to delete this occurrence")

        db.delete(occurrence)
        db.commit()
        return {"detail": "Activity occurrence deleted successfully"}
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.put("/activity-occurrences/{occurrence_id}/complete")
def complete_activity_occurrence(occurrence_id: int, db: Session = Depends(get_db)):
    """Mark an activity occurrence as complete"""
    try:
        # Validate occurrence_id
        if occurrence_id <= 0:
            return SecureErrorResponse.validation_error("Invalid occurrence ID")
        
        occurrence = db.query(ActivityOccurrence).filter(ActivityOccurrence.id == occurrence_id).first()
        if not occurrence:
            return SecureErrorResponse.not_found_error("Activity occurrence not found")
        
        # Mark as complete
        occurrence.complete = True
        db.commit()
        db.refresh(occurrence)
        
        return ActivityOccurrenceResponse(
            id=occurrence.id,
            activity_id=occurrence.activity_id,
            date=occurrence.date,
            complete=occurrence.complete,
            activity_title=occurrence.activity.title if occurrence.activity else None
        )
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.get("/activities/{activity_id}/occurrences")
def get_activity_occurrences(activity_id: int, db: Session = Depends(get_db)):
    """Obtener todas las ocurrencias de una actividad específica"""
    try:
        # Validate activity_id
        if activity_id <= 0:
            return SecureErrorResponse.validation_error("Invalid activity ID")
        
        activity = db.query(Activity).filter(Activity.id == activity_id).first()
        if not activity:
            return SecureErrorResponse.not_found_error("Activity not found")
        
        occurrences = db.query(ActivityOccurrence).filter(ActivityOccurrence.activity_id == activity_id).all()
        return [
            ActivityOccurrenceResponse(
                id=occ.id,
                activity_id=occ.activity_id,
                date=occ.date,
                complete=occ.complete,
                activity_title=activity.title
            )
            for occ in occurrences
        ]
    
    except Exception as e:
        return handle_generic_exception(e)

@app.get("/activity-occurrences/by-date/{date}")
def get_occurrences_by_date(date: str, db: Session = Depends(get_db)):
    """Obtener todas las ocurrencias para una fecha específica (formato: YYYY-MM-DD)"""
    try:
        # Validate and sanitize date parameter
        date_str = sanitize_string(date, 10)  # YYYY-MM-DD format
        
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return SecureErrorResponse.validation_error("Invalid date format. Use YYYY-MM-DD")
        
        # Validate date is not too far in the past or future (reasonable limits)
        from datetime import timedelta
        min_date = datetime.now().date() - timedelta(days=365 * 2)  # 2 years ago
        max_date = datetime.now().date() + timedelta(days=365 * 2)  # 2 years from now
        
        if target_date < min_date:
            return SecureErrorResponse.validation_error("Date is too far in the past")
        if target_date > max_date:
            return SecureErrorResponse.validation_error("Date is too far in the future")
        
        occurrences = db.query(ActivityOccurrence).filter(
            ActivityOccurrence.date >= datetime.combine(target_date, datetime.min.time()),
            ActivityOccurrence.date < datetime.combine(target_date, datetime.max.time())
        ).all()
        
        return [
            ActivityOccurrenceResponse(
                id=occ.id,
                activity_id=occ.activity_id,
                date=occ.date,
                complete=occ.complete,
                activity_title=occ.activity.title if occ.activity else None
            )
            for occ in occurrences
        ]
    
    except Exception as e:
        return handle_generic_exception(e)

@app.get("/activities/{activity_id}/todos")
def get_activity_todos(activity_id: int, db: Session = Depends(get_db)):
    try:
        # Validate activity_id
        if activity_id <= 0:
            return SecureErrorResponse.validation_error("Invalid activity ID")
        
        activity = db.query(Activity).filter(Activity.id == activity_id).first()
        if not activity:
            return SecureErrorResponse.not_found_error("Activity not found")
        return activity.todos
    
    except Exception as e:
        return handle_generic_exception(e)

@app.post("/activities/{activity_id}/todos")
def add_todo_to_activity(activity_id: int, todo: TodoCreate, db: Session = Depends(get_db)):
    try:
        # Validate activity_id
        if activity_id <= 0:
            return SecureErrorResponse.validation_error("Invalid activity ID")
        
        activity = db.query(Activity).filter(Activity.id == activity_id).first()
        if not activity:
            return SecureErrorResponse.not_found_error("Activity not found")
        
        # Additional validation
        if not todo.text.strip():
            return SecureErrorResponse.validation_error("Todo text cannot be empty")
        
        # Check for reasonable limit of todos per activity
        current_todo_count = db.query(Todo).filter(Todo.activity_id == activity_id).count()
        if current_todo_count >= 50:
            return SecureErrorResponse.validation_error("Maximum number of todos per activity reached (50)")
        
        db_todo = Todo(
            text=todo.text.strip(), 
            complete=todo.complete, 
            activity_id=activity_id
        )
        db.add(db_todo)
        db.commit()
        db.refresh(db_todo)
        return db_todo
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.get("/todos/{todo_id}")
def get_todo(todo_id: int, db: Session = Depends(get_db)):
    try:
        # Validate todo_id
        if todo_id <= 0:
            return SecureErrorResponse.validation_error("Invalid todo ID")
        
        todo = db.query(Todo).filter(Todo.id == todo_id).first()
        if not todo:
            return SecureErrorResponse.not_found_error("Todo not found")
        return todo
    
    except Exception as e:
        return handle_generic_exception(e)

@app.put("/todos/{todo_id}")
def update_todo(todo_id: int, todo_update: TodoUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Validate todo_id
        if todo_id <= 0:
            return SecureErrorResponse.validation_error("Invalid todo ID")
        
        todo = db.query(Todo).filter(Todo.id == todo_id).first()
        if not todo:
            return SecureErrorResponse.not_found_error("Todo not found")

        # Authorization: Only responsible users or admins can update
        if current_user not in todo.activity.responsibles and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to update this todo")
        
        update_data = todo_update.model_dump(exclude_unset=True)
        
        # Additional validation for text if being updated
        if "text" in update_data:
            if not update_data["text"].strip():
                return SecureErrorResponse.validation_error("Todo text cannot be empty")
            update_data["text"] = update_data["text"].strip()
        
        for key, value in update_data.items():
            setattr(todo, key, value)
        db.commit()
        db.refresh(todo)
        return todo
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Validate todo_id
        if todo_id <= 0:
            return SecureErrorResponse.validation_error("Invalid todo ID")
        
        todo = db.query(Todo).filter(Todo.id == todo_id).first()
        if not todo:
            return SecureErrorResponse.not_found_error("Todo not found")

        # Authorization: Only responsible users or admins can delete
        if current_user not in todo.activity.responsibles and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to delete this todo")
        
        db.delete(todo)
        db.commit()
        return {"detail": "Todo deleted successfully"}
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(History).order_by(History.timestamp.desc()).all()

@app.post("/history")
def create_history(history: HistoryCreate, db: Session = Depends(get_db)):
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == history.user_id).first()
        if not user:
            return SecureErrorResponse.not_found_error("User not found")
        
        # Additional validation
        if not history.action.strip():
            return SecureErrorResponse.validation_error("Action cannot be empty")
        
        db_history = History(
            action=history.action.strip(), 
            user_id=history.user_id
        )
        db.add(db_history)
        db.commit()
        db.refresh(db_history)
        return db_history
    
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle SQLAlchemy database errors securely"""
    error_id = SecureErrorResponse.generate_error_id()
    SecureErrorResponse.log_detailed_error(
        error_id=error_id,
        error=exc,
        request=request,
        additional_context={"error_category": "database", "error_type": type(exc).__name__}
    )
    if isinstance(exc, IntegrityError):
        if "UNIQUE constraint failed" in str(exc):
            return SecureErrorResponse.conflict_error("Resource already exists", error_id)
        elif "FOREIGN KEY constraint failed" in str(exc):
            return SecureErrorResponse.validation_error("Invalid reference data", error_id)
        else:
            return SecureErrorResponse.database_error("Data integrity error", error_id)
    return SecureErrorResponse.database_error("Database operation failed", error_id)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with sanitized messages"""
    error_id = SecureErrorResponse.generate_error_id()
    SecureErrorResponse.log_detailed_error(
        error_id=error_id,
        error=exc,
        request=request,
        additional_context={
            "error_category": "http_exception",
            "status_code": exc.status_code,
            "original_detail": exc.detail
        }
    )
    from error_handlers import sanitize_error_message
    sanitized_detail = sanitize_error_message(str(exc.detail))
    # Map status codes to error categories and error names
    error_map = {
        400: ("validation_error", SecureErrorResponse.validation_error),
        401: ("authentication_error", SecureErrorResponse.authentication_error),
        403: ("authorization_error", SecureErrorResponse.authorization_error),
        404: ("not_found_error", SecureErrorResponse.not_found_error),
        409: ("conflict_error", SecureErrorResponse.conflict_error),
        429: ("rate_limit_error", SecureErrorResponse.rate_limit_error),
    }
    error_name, error_func = error_map.get(exc.status_code, ("internal_server_error", SecureErrorResponse.internal_server_error))
    # Return a consistent error format with detail
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_name,
            "message": sanitized_detail,
            "detail": sanitized_detail,
            "error_id": error_id
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_id = SecureErrorResponse.generate_error_id()
    SecureErrorResponse.log_detailed_error(
        error_id=error_id,
        error=exc,
        request=request,
        additional_context={"error_category": "unexpected"}
    )
    return SecureErrorResponse.internal_server_error("An internal server error occurred", error_id)

@app.delete("/activities/{activity_id}")
def delete_activity(activity_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        # Validate activity_id
        if activity_id <= 0:
            return SecureErrorResponse.validation_error("Invalid activity ID")
        activity = db.query(Activity).filter(Activity.id == activity_id).first()
        if not activity:
            return SecureErrorResponse.not_found_error("Activity not found")
        # Authorization: Only responsible users or admins can delete
        if current_user not in activity.responsibles and not current_user.is_admin:
            return SecureErrorResponse.authorization_error("Not authorized to delete this activity")
        db.delete(activity)
        db.commit()
        return {"detail": "Activity deleted successfully"}
    except Exception as e:
        db.rollback()
        return handle_database_error(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10242, reload=True)

