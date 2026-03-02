from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import models
from .database import engine

# Automatically create database tables during startup
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Antigravity API", version="1.0.0")

# Configure CORS to allow any origin for now (for Lovable frontend development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Antigravity API. The engine is running."}

# Routers to be included below:
from .routers import auth
app.include_router(auth.router, prefix="/auth", tags=["auth"])
from .routers import analysis
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
from .routers import inventory, forecast, bom
app.include_router(inventory.router, prefix="/inventory", tags=["inventory"])
app.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
app.include_router(bom.router, prefix="/bom", tags=["bom"])
