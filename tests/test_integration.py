import pytest
from fastapi.testclient import TestClient
from main import app, Base, db_engine
from datetime import datetime, timezone

@pytest.fixture(autouse=True)
def setup_db():
    # Limpia y crea tablas antes de cada prueba
    Base.metadata.drop_all(bind=db_engine)
    Base.metadata.create_all(bind=db_engine)
    yield
    Base.metadata.drop_all(bind=db_engine)


@pytest.fixture
def client():
    return TestClient(app)
 
def test_full_user_category_activity_flow(client):
    # Crear usuario
    resp_user = client.post("/users", json={
        "name": "Iñigo", "username": "inigo", "password": "Password123!", "is_admin": False
    })
    assert resp_user.status_code == 200
    user_id = resp_user.json()["id"]

    # Login y generar token
    token = resp_user.json().get("id")  # no auth en endpoints salvo token

    # Crear categoría
    resp_cat = client.post("/categories", json={
        "name": "Work", "icon_name": "briefcase", "mode": "work"
    })
    assert resp_cat.status_code == 200
    cat_id = resp_cat.json()["id"]

    # Crear actividad
    payload_act = {
        "title": "Planificación", "start_date": datetime.now(timezone.utc).isoformat(),
        "time": "09:00", "category_id": cat_id,
        "repeat_mode": "none", "end_date": None,
        "days_of_week": [], "day_of_month": None,
        "notes": "Reunión semanal", "mode": "work",
        "responsible_ids": [user_id], "todos": []
    }
    resp_act = client.post("/activities", json=payload_act)
    assert resp_act.status_code == 200
    act_id = resp_act.json()["id"]

    # Obtener actividad
    resp_get = client.get(f"/activities/{act_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["title"] == payload_act["title"]

    # Actualizar actividad
    resp_upd = client.put(f"/activities/{act_id}", json={"notes": "Actualizado"})
    assert resp_upd.status_code == 200
    assert resp_upd.json()["notes"] == "Actualizado"

    # Eliminar actividad
    resp_del = client.delete(f"/activities/{act_id}")
    assert resp_del.status_code == 200
    # Confirmar eliminación
    resp_404 = client.get(f"/activities/{act_id}")
    assert resp_404.status_code == 404

def test_todo_and_history_flow(client):
    # Crear usuario y categoría
    resp_user = client.post("/users", json={
        "name": "Lola", "username": "lola", "password": "Password456!", "is_admin": False
    })
    user_id = resp_user.json()["id"]
    resp_cat = client.post("/categories", json={
        "name": "Personal", "icon_name": "user", "mode": "personal"
    })
    cat_id = resp_cat.json()["id"]

    # Crear actividad
    payload_act = {
        "title": "Comprar leche", "start_date": datetime.now(timezone.utc).isoformat(),
        "time": "18:00", "category_id": cat_id,
        "repeat_mode": "none", "end_date": None,
        "days_of_week": [], "day_of_month": None,
        "notes": None, "mode": "personal",
        "responsible_ids": [], "todos": []
    }
    resp_act = client.post("/activities", json=payload_act)
    act_id = resp_act.json()["id"]

    # Agregar todo
    resp_todo = client.post(f"/activities/{act_id}/todos", json={
        "text": "Ir a la tienda", "complete": False
    })
    assert resp_todo.status_code == 200
    todo_id = resp_todo.json()["id"]

    # Obtener todos de la actividad
    resp_list = client.get(f"/activities/{act_id}/todos")
    assert resp_list.status_code == 200
    assert any(t["id"] == todo_id for t in resp_list.json())

    # Crear historial
    resp_hist = client.post("/history", json={
        "action": "Creó todo", "user_id": user_id
    })
    assert resp_hist.status_code == 200
    hist_id = resp_hist.json()["id"]

    # Obtener historial
    resp_his_list = client.get("/history")
    assert resp_his_list.status_code == 200
    assert any(h["id"] == hist_id for h in resp_his_list.json())
