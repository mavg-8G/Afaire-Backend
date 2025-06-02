from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Form
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
db_engine = create_engine("sqlite:///./todo_app.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)

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
    name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

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
    todos = relationship("Todo", back_populates="activity")

class Todo(Base):
    __tablename__ = 'todos'
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    activity_id = Column(Integer, ForeignKey("activities.id"), nullable=False)
    activity = relationship("Activity", back_populates="todos")

class History(Base):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User")

Base.metadata.create_all(bind=db_engine)

# Schemas
class UserCreate(BaseModel):
    name: str
    username: str
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

class CategoryCreate(BaseModel):
    name: str
    icon_name: str
    mode: CategoryMode

class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    icon_name: Optional[str] = None
    mode: Optional[CategoryMode] = None

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

class ActivityUpdate(BaseModel):
    title: Optional[str]
    start_date: Optional[datetime]
    time: Optional[str]
    category_id: Optional[int]
    repeat_mode: Optional[RepeatMode]
    end_date: Optional[datetime]
    days_of_week: Optional[List[str]]
    day_of_month: Optional[int]
    notes: Optional[str]
    mode: Optional[CategoryMode]
    responsible_ids: Optional[List[int]] = []
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
@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Verificar que el username no esté en uso
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    db_user = User(
        name=user.name,
        username=user.username,
        hashed_password=get_password_hash(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.put("/users/{user_id}")
def update_user(user_id: int, name: Optional[str] = Form(None), username: Optional[str] = Form(None), password: Optional[str] = Form(None), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validar que el username no esté en uso por otro usuario
    if username is not None:
        existing_user = db.query(User).filter(User.username == username, User.id != user_id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
    
    if name is not None:
        user.name = name
    if username is not None:
        user.username = username
    if password is not None:
        user.hashed_password = get_password_hash(password)
    
    db.commit()
    db.refresh(user)
    return user

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

@app.post("/categories")
def create_category(category: CategoryCreate, db: Session = Depends(get_db)):
    db_category = Category(**category.dict())
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category

@app.get("/categories")
def get_categories(db: Session = Depends(get_db)):
    return db.query(Category).all()

@app.get("/categories/{category_id}")
def get_category(category_id: int, db: Session = Depends(get_db)):
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category

@app.put("/categories/{category_id}")
def update_category(category_id: int, category_update: CategoryUpdate, db: Session = Depends(get_db)):
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    update_data = category_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(category, key, value)
    db.commit()
    db.refresh(category)
    return category

@app.delete("/categories/{category_id}")
def delete_category(category_id: int, db: Session = Depends(get_db)):
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    db.delete(category)
    db.commit()
    return {"detail": "Category deleted"}

@app.get("/activities")
def get_activities(db: Session = Depends(get_db)):
    return db.query(Activity).all()


@app.post("/activities")
def create_activity(activity: ActivityCreate, db: Session = Depends(get_db)):
    # Verificar que la categoría existe
    category = db.query(Category).filter(Category.id == activity.category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Verificar que todos los usuarios responsables existen
    if activity.responsible_ids:
        existing_users = db.query(User).filter(User.id.in_(activity.responsible_ids)).all()
        if len(existing_users) != len(activity.responsible_ids):
            raise HTTPException(status_code=404, detail="One or more users not found")
    
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
    
    if activity.responsible_ids:
        db_activity.responsibles = existing_users
    
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)

    for todo in activity.todos:
        db_todo = Todo(text=todo.text, activity_id=db_activity.id)
        db.add(db_todo)

    db.commit()
    return db_activity

@app.get("/activities/{activity_id}")
def get_activity(activity_id: int, db: Session = Depends(get_db)):
    activity = db.query(Activity).filter(Activity.id == activity_id).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    return activity

@app.put("/activities/{activity_id}")
def update_activity(activity_id: int, activity_update: ActivityUpdate, db: Session = Depends(get_db)):
    activity = db.query(Activity).filter(Activity.id == activity_id).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")

    update_data = activity_update.dict(exclude_unset=True)
    
    # Validar category_id si se está actualizando
    if "category_id" in update_data:
        category = db.query(Category).filter(Category.id == update_data["category_id"]).first()
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")

    if "days_of_week" in update_data and update_data["days_of_week"] is not None:
        update_data["days_of_week"] = ",".join(update_data["days_of_week"])

    for key, value in update_data.items():
        if key == "responsible_ids":
            if value:  # Solo validar si hay IDs
                existing_users = db.query(User).filter(User.id.in_(value)).all()
                if len(existing_users) != len(value):
                    raise HTTPException(status_code=404, detail="One or more users not found")
                activity.responsibles = existing_users
            else:
                activity.responsibles = []
        else:
            setattr(activity, key, value)

    db.commit()
    db.refresh(activity)
    return activity

@app.delete("/activities/{activity_id}")
def delete_activity(activity_id: int, db: Session = Depends(get_db)):
    activity = db.query(Activity).filter(Activity.id == activity_id).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    db.delete(activity)
    db.commit()
    return {"detail": "Activity deleted successfully"}

@app.get("/activities/{activity_id}/todos")
def get_activity_todos(activity_id: int, db: Session = Depends(get_db)):
    activity = db.query(Activity).filter(Activity.id == activity_id).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    return activity.todos

@app.post("/activities/{activity_id}/todos")
def add_todo_to_activity(activity_id: int, todo: TodoCreate, db: Session = Depends(get_db)):
    activity = db.query(Activity).filter(Activity.id == activity_id).first()
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    db_todo = Todo(text=todo.text, activity_id=activity_id)
    db.add(db_todo)
    db.commit()
    db.refresh(db_todo)
    return db_todo

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int, db: Session = Depends(get_db)):
    todo = db.query(Todo).filter(Todo.id == todo_id).first()
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    db.delete(todo)
    db.commit()
    return {"detail": "Todo deleted successfully"}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(History).order_by(History.timestamp.desc()).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10242, reload=True)

