from backend.database import SessionLocal, engine
from backend import models, utils

def init_db():
    db = SessionLocal()
    
    # Create roles if they don't exist
    admin_role = db.query(models.Role).filter(models.Role.name == "System Admin").first()
    if not admin_role:
        admin_role = models.Role(name="System Admin", permissions={"all": True})
        db.add(admin_role)
    
    sales_role = db.query(models.Role).filter(models.Role.name == "Sales Manager").first()
    if not sales_role:
        sales_role = models.Role(name="Sales Manager", permissions={"sales": True})
        db.add(sales_role)
        
    db.commit()
    
    # Create default admin user
    admin_user = db.query(models.User).filter(models.User.email == "admin@navonmesh.com").first()
    if not admin_user:
        hashed_password = utils.get_password_hash("admin123")
        admin_user = models.User(
            name="Super Admin",
            email="admin@navonmesh.com",
            password_hash=hashed_password,
            role_id=admin_role.id
        )
        db.add(admin_user)
        db.commit()
        print("Created default admin user: admin@navonmesh.com / admin123")
    else:
        print("Admin user already exists.")
        
    db.close()

if __name__ == "__main__":
    init_db()
