from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Response
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date, time
from enum import Enum
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Table, Text, Boolean, Enum as SqlEnum
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from VerificarToken import create_access_token, verify_token

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
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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
    todos = relationship("Todo", back_populates="activity")

class ActivityOccurrence(Base):
    __tablename__ = 'activity_occurrences'
    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activities.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    complete = Column(Boolean, default=False, nullable=False)
    activity = relationship("Activity", backref="occurrences")

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

Base.metadata.create_all(bind=db_engine)

# Schemas
class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class UserCreate(BaseModel):
    name: str
    username: str
    password: str
    is_admin: bool = False 

class UserUpdate(BaseModel):
    name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    is_admin: Optional[bool] = None

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
    complete: bool = False

class TodoUpdate(BaseModel):
    text: Optional[str] = None
    complete: Optional[bool] = None

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

    class Config:
        orm_mode = True

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

class HistoryCreate(BaseModel):
    action: str
    user_id: int
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
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    print(f"Login attempt for username: {form_data.username}")  # Debug log
    
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user:
        print(f"User not found: {form_data.username}")  # Debug log
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    print(f"User found: {user.username}")  # Debug log
    password_valid = pwd_context.verify(form_data.password, user.hashed_password)
    print(f"Password valid: {password_valid}")  # Debug log
    
    if not password_valid:
        print(f"Invalid password for user: {user.username}")  # Debug log
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    try:
        access_token = create_access_token(data={"sub": str(user.id)})
        print(f"Token created successfully for user: {user.username}")  # Debug log
        print(f"Token: {access_token[:50]}...")  # Debug log - mostrar solo parte del token
        
        response = {
            "access_token": access_token, 
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username,
            "is_admin": user.is_admin
        }
        print(f"Login successful for user: {user.username}")  # Debug log
        return response
    except Exception as e:
        print(f"Error creating token: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail="Error creating authentication token")


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
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    db_user = User(
        name=user.name,
        username=user.username,
        hashed_password=get_password_hash(user.password),
        is_admin=user.is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.put("/users/{user_id}")
def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_update.username is not None:
        existing_user = db.query(User).filter(User.username == user_update.username, User.id != user_id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
    
    if user_update.name is not None:
        user.name = user_update.name
    if user_update.username is not None:
        user.username = user_update.username
    if user_update.password is not None:
        user.hashed_password = get_password_hash(user_update.password)
    if user_update.is_admin is not None:
        user.is_admin = user_update.is_admin  # Update is_admin
    
    db.commit()
    db.refresh(user)
    return user

@app.post("/users/{user_id}/change-password")
def change_password(
    user_id: int,
    req: ChangePasswordRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not pwd_context.verify(req.old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    user.hashed_password = get_password_hash(req.new_password)
    db.commit()
    return {"detail": "Password updated successfully"}


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
    # Verificar que la categoría existe
    category = db.query(Category).filter(Category.id == activity.category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Verificar que todos los usuarios responsables existen
    existing_users = []
    if activity.responsible_ids:
        existing_users = db.query(User).filter(User.id.in_(activity.responsible_ids)).all()
        if len(existing_users) != len(activity.responsible_ids):
            raise HTTPException(status_code=404, detail="One or more users not found")
    
    # Crear la actividad
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
    
    # Asignar usuarios responsables
    if existing_users:
        db_activity.responsibles = existing_users
    
    # Guardar la actividad primero
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)  # Importante: obtener el ID generado
    
    # Agregar TODOs después de que la actividad tenga un ID
    for todo_data in activity.todos:
        db_todo = Todo(
            text=todo_data.text, 
            complete=todo_data.complete,
            activity_id=db_activity.id
        )
        db.add(db_todo)
    
    # Commit final para los TODOs
    db.commit()
    
    # Refresh final para obtener la actividad con todas sus relaciones
    db.refresh(db_activity)
    
    print(f"Activity created successfully with ID: {db_activity.id}")  # Debug log
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

@app.put("/activity-occurrences/{occurrence_id}/complete")
def complete_occurrence(occurrence_id: int, db: Session = Depends(get_db)):
    occurrence = db.query(ActivityOccurrence).filter(ActivityOccurrence.id == occurrence_id).first()
    if not occurrence:
        raise HTTPException(status_code=404, detail="Occurrence not found")
    occurrence.complete = True
    db.commit()
    db.refresh(occurrence)
    return occurrence

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
    db_todo = Todo(text=todo.text, complete=todo.complete, activity_id=activity_id)
    db.add(db_todo)
    db.commit()
    db.refresh(db_todo)
    return db_todo

@app.get("/todos/{todo_id}")
def get_todo(todo_id: int, db: Session = Depends(get_db)):
    todo = db.query(Todo).filter(Todo.id == todo_id).first()
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo

@app.put("/todos/{todo_id}")
def update_todo(todo_id: int, todo_update: TodoUpdate, db: Session = Depends(get_db)):
    todo = db.query(Todo).filter(Todo.id == todo_id).first()
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    update_data = todo_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(todo, key, value)
    db.commit()
    db.refresh(todo)
    return todo

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

@app.post("/history")
def create_history(history: HistoryCreate, db: Session = Depends(get_db)):
    # Verifica que el usuario existe
    user = db.query(User).filter(User.id == history.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db_history = History(action=history.action, user_id=history.user_id)
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10242, reload=True)

