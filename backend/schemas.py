from pydantic import BaseModel, EmailStr
from typing import Optional, List, Any, Dict
from datetime import datetime

class RoleBase(BaseModel):
    name: str
    permissions: Dict[str, Any] = {}

class Role(RoleBase):
    id: int

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserCreate(UserBase):
    password: str
    role_id: int

class User(UserBase):
    id: int
    role_id: int

    class Config:
        from_attributes = True

class ProductBase(BaseModel):
    sku: str
    name: str
    current_stock: float = 0.0
    safety_stock: float = 0.0

class ProductCreate(ProductBase):
    pass

class Product(ProductBase):
    id: int

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class HistoricalAnalysisBase(BaseModel):
    forecast_parameters: Dict[str, Any]
    result_json: Dict[str, Any]

class HistoricalAnalysisCreate(HistoricalAnalysisBase):
    pass

class HistoricalAnalysis(HistoricalAnalysisBase):
    id: int
    user_id: int
    date_run: datetime

    class Config:
        from_attributes = True
