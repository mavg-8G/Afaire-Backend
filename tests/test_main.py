import pytest
from fastapi.testclient import TestClient
from main import app, Base, db_engine, SessionLocal, get_password_hash
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from main import User
from main import Category, CategoryMode
from main import ActivityOccurrence
from main import History

# test_main.py


client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    # Recreate all tables before each test
    Base.metadata.drop_all(bind=db_engine)
    Base.metadata.create_all(bind=db_engine)
    yield
    # Drop all tables after each test
    Base.metadata.drop_all(bind=db_engine)

def create_user_helper(username="testuser", password="TestPass123!", is_admin=False):
    db: Session = SessionLocal()
    user = User(
        name="Test User",
        username=username,
        hashed_password=get_password_hash(password),
        is_admin=is_admin
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user

def create_category_helper(name="TestCat", icon_name="icon", mode="personal"):
    db: Session = SessionLocal()
    category = Category(
        name=name,
        icon_name=icon_name,
        mode=CategoryMode(mode)
    )
    db.add(category)
    db.commit()
    db.refresh(category)
    db.close()
    return category

def create_activity_helper(category_id, responsible_ids=None):
    if responsible_ids is None:
        responsible_ids = []
    payload = {
        "title": "Test Activity",
        "start_date": datetime.now(timezone.utc).isoformat(),
        "time": "12:00",
        "category_id": category_id,
        "repeat_mode": "none",
        "end_date": None,
        "days_of_week": [],
        "day_of_month": None,
        "notes": "Some notes",
        "mode": "personal",
        "responsible_ids": responsible_ids,
        "todos": []
    }
    response = client.post("/activities", json=payload)
    return response

def get_token(username="testuser", password="TestPass123!"):
    response = client.post("/token", data={"username": username, "password": password})
    return response.json().get("access_token")

# --- /token ---
def test_login_success():
    create_user_helper()
    response = client.post("/token", data={"username": "testuser", "password": "TestPass123!"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}, response: {response.text}"
    assert "access_token" in response.json(), f"Response JSON: {response.json()}"

def test_login_failure_wrong_password():
    create_user_helper()
    response = client.post("/token", data={"username": "testuser", "password": "wrong"})
    assert response.status_code == 401, f"Expected 401, got {response.status_code}, response: {response.text}"

def test_login_failure_no_user():
    response = client.post("/token", data={"username": "nouser", "password": "pass"})
    assert response.status_code == 401, f"Expected 401, got {response.status_code}, response: {response.text}"

# --- /users ---
def test_create_user_success():
    response = client.post("/users", json={
        "name": "User1", "username": "user1", "password": "Password123!", "is_admin": False
    })
    assert response.status_code == 200
    assert response.json()["username"] == "user1"

def test_create_user_duplicate_username():
    create_user_helper(username="user2")
    response = client.post("/users", json={
        "name": "User2", "username": "user2", "password": "Password123!", "is_admin": False
    })
    assert response.status_code in (400, 409), f"Expected 400 or 409, got {response.status_code}, response: {response.text}"

def test_get_users():
    create_user_helper(username="user3")
    response = client.get("/users")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_user_success():
    user = create_user_helper(username="user4")
    response = client.get(f"/users/{user.id}")
    assert response.status_code == 200
    assert response.json()["username"] == "user4"

def test_get_user_not_found():
    response = client.get("/users/999")
    assert response.status_code == 404

def test_update_user_success():
    user = create_user_helper(username="user5")
    response = client.put(f"/users/{user.id}", json={"name": "Updated", "is_admin": True})
    assert response.status_code == 200
    assert response.json()["name"] == "Updated"
    assert response.json()["is_admin"] is True

def test_update_user_username_exists():
    user1 = create_user_helper(username="user6")
    user2 = create_user_helper(username="user7")
    response = client.put(f"/users/{user2.id}", json={"username": "user6"})
    assert response.status_code in (400, 409), f"Expected 400 or 409, got {response.status_code}, response: {response.text}"

def test_update_user_not_found():
    response = client.put("/users/999", json={"name": "Nope"})
    assert response.status_code == 404

def test_change_password_success():
    user = create_user_helper(username="user8", password="OldPass123!")
    response = client.post(f"/users/{user.id}/change-password", json={
        "old_password": "OldPass123!", "new_password": "NewPass456@"
    })
    assert response.status_code == 200

def test_change_password_wrong_old():
    user = create_user_helper(username="user9", password="OldPass123!")
    response = client.post(f"/users/{user.id}/change-password", json={
        "old_password": "WrongPass123!", "new_password": "NewPass456@"
    })
    assert response.status_code == 401

def test_change_password_user_not_found():
    response = client.post("/users/999/change-password", json={
        "old_password": "OldPass123!", "new_password": "NewPass456@"
    })
    assert response.status_code == 404

def test_delete_user_success():
    user = create_user_helper(username="user10")
    response = client.delete(f"/users/{user.id}")
    assert response.status_code == 200

def test_delete_user_not_found():
    response = client.delete("/users/999")
    assert response.status_code == 404

# --- /categories ---
def test_create_category_success():
    response = client.post("/categories", json={
        "name": "Cat1", "icon_name": "icon", "mode": "personal"
    })
    assert response.status_code == 200
    assert response.json()["name"] == "Cat1"

def test_get_categories():
    create_category_helper(name="Cat2")
    response = client.get("/categories")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_category_success():
    cat = create_category_helper(name="Cat3")
    response = client.get(f"/categories/{cat.id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Cat3"

def test_get_category_not_found():
    response = client.get("/categories/999")
    assert response.status_code == 404

def test_update_category_success():
    cat = create_category_helper(name="Cat4")
    response = client.put(f"/categories/{cat.id}", json={"name": "UpdatedCat"})
    assert response.status_code == 200
    assert response.json()["name"] == "UpdatedCat"

def test_update_category_not_found():
    response = client.put("/categories/999", json={"name": "NoCat"})
    assert response.status_code == 404

def test_delete_category_success():
    cat = create_category_helper(name="Cat5")
    response = client.delete(f"/categories/{cat.id}")
    assert response.status_code == 200

def test_delete_category_not_found():
    response = client.delete("/categories/999")
    assert response.status_code == 404

# --- /activities ---
def test_create_activity_success():
    user = create_user_helper(username="user11")
    cat = create_category_helper(name="Cat6")
    response = create_activity_helper(category_id=cat.id, responsible_ids=[user.id])
    assert response.status_code == 200
    assert response.json()["title"] == "Test Activity"

def test_create_activity_category_not_found():
    response = create_activity_helper(category_id=999)
    assert response.status_code == 404

def test_create_activity_user_not_found():
    cat = create_category_helper(name="Cat7")
    response = create_activity_helper(category_id=cat.id, responsible_ids=[999])
    assert response.status_code == 404

def test_get_activities():
    cat = create_category_helper(name="Cat8")
    create_activity_helper(category_id=cat.id)
    response = client.get("/activities")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_activity_success():
    cat = create_category_helper(name="Cat9")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    response = client.get(f"/activities/{activity_id}")
    assert response.status_code == 200
    assert response.json()["id"] == activity_id

def test_get_activity_not_found():
    response = client.get("/activities/999")
    assert response.status_code == 404

def test_update_activity_success():
    user = create_user_helper(username="user12")
    cat = create_category_helper(name="Cat10")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    response = client.put(f"/activities/{activity_id}", json={
        "title": "Updated Activity", "responsible_ids": [user.id]
    })
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Activity"

def test_update_activity_not_found():
    response = client.put("/activities/999", json={"title": "Nope"})
    assert response.status_code == 404

def test_update_activity_category_not_found():
    cat = create_category_helper(name="Cat11")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    response = client.put(f"/activities/{activity_id}", json={"category_id": 999})
    assert response.status_code == 404

def test_update_activity_user_not_found():
    cat = create_category_helper(name="Cat12")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    response = client.put(f"/activities/{activity_id}", json={"responsible_ids": [999]})
    assert response.status_code == 404

def test_delete_activity_success():
    cat = create_category_helper(name="Cat13")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    response = client.delete(f"/activities/{activity_id}")
    assert response.status_code == 200

def test_delete_activity_not_found():
    response = client.delete("/activities/999")
    assert response.status_code == 404

# --- /activity-occurrences/{occurrence_id}/complete ---
def test_complete_occurrence_success():
    # Setup: create activity and occurrence
    cat = create_category_helper(name="Cat14")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    db: Session = SessionLocal()
    occ = ActivityOccurrence(activity_id=activity_id, date=datetime.now(timezone.utc))
    db.add(occ)
    db.commit()
    db.refresh(occ)
    db.close()
    response = client.put(f"/activity-occurrences/{occ.id}/complete")
    assert response.status_code == 200
    assert response.json()["complete"] is True

def test_complete_occurrence_not_found():
    response = client.put("/activity-occurrences/999/complete")
    assert response.status_code == 404

# --- /activities/{activity_id}/todos ---
def test_get_activity_todos_success():
    cat = create_category_helper(name="Cat15")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    response = client.get(f"/activities/{activity_id}/todos")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_activity_todos_not_found():
    response = client.get("/activities/999/todos")
    assert response.status_code == 404

def test_add_todo_to_activity_success():
    cat = create_category_helper(name="Cat16")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    response = client.post(f"/activities/{activity_id}/todos", json={"text": "Todo1", "complete": False})
    assert response.status_code == 200
    assert response.json()["text"] == "Todo1"

def test_add_todo_to_activity_not_found():
    response = client.post("/activities/999/todos", json={"text": "Todo2", "complete": False})
    assert response.status_code == 404

# --- /todos/{todo_id} ---
def test_get_todo_success():
    cat = create_category_helper(name="Cat17")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    todo_resp = client.post(f"/activities/{activity_id}/todos", json={"text": "Todo3", "complete": False})
    todo_id = todo_resp.json()["id"]
    response = client.get(f"/todos/{todo_id}")
    assert response.status_code == 200
    assert response.json()["text"] == "Todo3"

def test_get_todo_not_found():
    response = client.get("/todos/999")
    assert response.status_code == 404

def test_update_todo_success():
    cat = create_category_helper(name="Cat18")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    todo_resp = client.post(f"/activities/{activity_id}/todos", json={"text": "Todo4", "complete": False})
    todo_id = todo_resp.json()["id"]
    response = client.put(f"/todos/{todo_id}", json={"text": "Updated Todo", "complete": True})
    assert response.status_code == 200
    assert response.json()["text"] == "Updated Todo"
    assert response.json()["complete"] is True

def test_update_todo_not_found():
    response = client.put("/todos/999", json={"text": "Nope"})
    assert response.status_code == 404

def test_delete_todo_success():
    cat = create_category_helper(name="Cat19")
    resp = create_activity_helper(category_id=cat.id)
    activity_id = resp.json()["id"]
    todo_resp = client.post(f"/activities/{activity_id}/todos", json={"text": "Todo5", "complete": False})
    todo_id = todo_resp.json()["id"]
    response = client.delete(f"/todos/{todo_id}")
    assert response.status_code == 200

def test_delete_todo_not_found():
    response = client.delete("/todos/999")
    assert response.status_code == 404

# --- /history ---
def test_get_history():
    user = create_user_helper(username="user13")
    db: Session = SessionLocal()
    hist = History(action="test", user_id=user.id)
    db.add(hist)
    db.commit()
    db.close()
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_history_success():
    user = create_user_helper(username="user14")
    response = client.post("/history", json={"action": "did something", "user_id": user.id})
    assert response.status_code == 200
    assert response.json()["action"] == "did something"

def test_create_history_user_not_found():
    response = client.post("/history", json={"action": "fail", "user_id": 999})
    assert response.status_code == 404