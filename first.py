from main import SessionLocal, User, get_password_hash

def create_user():
    db = SessionLocal()
    try:
        username = "juntsPer"
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            print("User already exists.")
            return
        user = User(
            name="Matias",
            username=username,
            hashed_password=get_password_hash("Pruebas123!"),
            is_admin=True
        )
        db.add(user)
        db.commit()
        print("User created successfully!")
    except Exception as e:
        db.rollback()
        print(f"Error creating user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_user()