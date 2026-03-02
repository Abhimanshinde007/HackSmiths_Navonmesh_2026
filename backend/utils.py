from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError
import os

# Secret key should be loaded from env, hardcoded for now just for development
SECRET_KEY = os.getenv("SECRET_KEY", "b3c58be79f1df02beec8b1114532fe1d58ab66cb47228a6fca10ad350e9a7217")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60 # 30 days for MSME platform persistence convenience

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
