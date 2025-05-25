from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date, time
from enum import Enum
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Table, Text, Boolean, Enum as SqlEnum
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
from passlib.context import CryptContext

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()
db_engine = create_engine("sqlite:///./todo_app.db")
SessionLocal = sessionmaker(bind=db_engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
    name = Column(String)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    icon_name = Column(String)
    mode = Column(SqlEnum(CategoryMode))

class Activity(Base):
    __tablename__ = 'activities'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    start_date = Column(DateTime)
    time = Column(String)
    category_id = Column(Integer, ForeignKey("categories.id"))
    repeat_mode = Column(SqlEnum(RepeatMode), default=RepeatMode.none)
    end_date = Column(DateTime, nullable=True)
    days_of_week = Column(String, nullable=True)  # comma-separated: "mon,tue"
    day_of_month = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    mode = Column(SqlEnum(CategoryMode))
    category = relationship("Category")
    responsibles = relationship("User", secondary=activity_user, backref="activities")
    todos = relationship("Todo", back_populates="activity")

class Todo(Base):
    __tablename__ = 'todos'
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    activity_id = Column(Integer, ForeignKey("activities.id"))
    activity = relationship("Activity", back_populates="todos")

class History(Base):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User")

Base.metadata.create_all(bind=db_engine)

# Schemas
class UserCreate(BaseModel):
    name: str
    username: str
    password: str

class CategoryCreate(BaseModel):
    name: str
    icon_name: str
    mode: CategoryMode

class TodoCreate(BaseModel):
    text: str

class ActivityCreate(BaseModel):
    title: str
    start_date: datetime
    time: str
    category_id: int
    repeat_mode: RepeatMode = RepeatMode.none
    end_date: Optional[datetime] = None
    days_of_week: Optional[List[str]] = None
    day_of_month: Optional[int] = None
    notes: Optional[str] = None
    mode: CategoryMode
    responsible_ids: List[int] = []
    todos: List[TodoCreate] = []

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
@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(
        name=user.name,
        username=user.username,
        hashed_password=get_password_hash(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/categories")
def create_category(category: CategoryCreate, db: Session = Depends(get_db)):
    db_category = Category(**category.dict())
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category

@app.post("/activities")
def create_activity(activity: ActivityCreate, db: Session = Depends(get_db)):
    db_activity = Activity(
        title=activity.title,
        start_date=activity.start_date,
        time=activity.time,
        category_id=activity.category_id,
        repeat_mode=activity.repeat_mode,
        end_date=activity.end_date,
        days_of_week=','.join(activity.days_of_week or []),
        day_of_month=activity.day_of_month,
        notes=activity.notes,
        mode=activity.mode
    )
    db_activity.responsibles = db.query(User).filter(User.id.in_(activity.responsible_ids)).all()
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)

    for todo in activity.todos:
        db_todo = Todo(text=todo.text, activity_id=db_activity.id)
        db.add(db_todo)

    db.commit()
    return db_activity

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(History).order_by(History.timestamp.desc()).all()
