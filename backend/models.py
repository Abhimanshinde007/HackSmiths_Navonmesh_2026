from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from .database import Base

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    permissions = Column(JSON, default={})

    users = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role_id = Column(Integer, ForeignKey("roles.id"))

    role = relationship("Role", back_populates="users")
    historical_analyses = relationship("HistoricalAnalysis", back_populates="user")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    current_stock = Column(Float, default=0.0)
    safety_stock = Column(Float, default=0.0)
    
    # We could set up relationships for BOM if needed later:
    # bom_components = relationship("BillOfMaterial", foreign_keys="[BillOfMaterial.parent_product_id]")
    # used_in = relationship("BillOfMaterial", foreign_keys="[BillOfMaterial.component_product_id]")


class BOMRecord(Base):
    __tablename__ = "bom_records"

    id = Column(Integer, primary_key=True, index=True)
    product = Column(String, unique=True, index=True)
    copper_type = Column(String)
    copper_weight_kg = Column(Float)
    lamination_type = Column(String)
    lamination_weight_kg = Column(Float)
    bobbin_type = Column(String)
    other_reqs = Column(String)


class HistoricalAnalysis(Base):
    __tablename__ = "historical_analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date_run = Column(DateTime)
    forecast_parameters = Column(JSON)
    result_json = Column(JSON)

    user = relationship("User", back_populates="historical_analyses")
