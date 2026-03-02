from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import io

from .. import database, models, schemas
from .auth import get_current_user

# Import the existing rock-solid business logic
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_processor import (
    ingest_sales_excel,
    ingest_purchase_excel,
    ingest_inward_excel,
    ingest_outward_excel,
    combine_stock_registers,
    compute_stock,
    get_anchor_customers,
    predict_reorder
)

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload/sales")
async def upload_sales(
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user)
):
    # Read files into memory for the data processor
    file_objs = []
    for f in files:
        content = await f.read()
        file_obj = io.BytesIO(content)
        file_obj.name = f.filename
        file_objs.append(file_obj)
        
    df, errors = ingest_sales_excel(file_objs)
    if errors and df.empty:
        raise HTTPException(status_code=400, detail="|".join(errors))
        
    # Convert dataframe to records for JSON response
    # We replace NaNs with None for valid JSON serialization
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return {"message": "Sales data ingested successfully", "errors": errors, "row_count": len(df), "data": records}

@router.post("/upload/purchase")
async def upload_purchase(
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user)
):
    file_objs = []
    for f in files:
        content = await f.read()
        file_obj = io.BytesIO(content)
        file_obj.name = f.filename
        file_objs.append(file_obj)
        
    df, errors = ingest_purchase_excel(file_objs)
    if errors and df.empty:
        raise HTTPException(status_code=400, detail="|".join(errors))
        
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return {"message": "Purchase data ingested successfully", "errors": errors, "row_count": len(df), "data": records}

@router.post("/upload/stock")
async def upload_stock_registers(
    inward_files: List[UploadFile] = File(None),
    outward_files: List[UploadFile] = File(None),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    in_objs = []
    if inward_files:
        for f in inward_files:
            content = await f.read()
            obj = io.BytesIO(content)
            obj.name = f.filename
            in_objs.append(obj)
            
    out_objs = []
    if outward_files:
        for f in outward_files:
            content = await f.read()
            obj = io.BytesIO(content)
            obj.name = f.filename
            out_objs.append(obj)

    in_df, in_errs = ingest_inward_excel(in_objs) if in_objs else (pd.DataFrame(), [])
    out_df, out_errs = ingest_outward_excel(out_objs) if out_objs else (pd.DataFrame(), [])
    
    combined = combine_stock_registers(in_df, out_df)
    if combined.empty:
        raise HTTPException(status_code=400, detail="No readable stock data found")
        
    stock_df, s_err = compute_stock(combined)
    
    # Store stock updates into the PostgreSQL Products table
    # This marries the Hackathon pandas logic with the new MSME Database
    for _, row in stock_df.iterrows():
        sku = str(row['material']).strip()
        qty = float(row['current_stock'])
        
        db_product = db.query(models.Product).filter(models.Product.sku == sku).first()
        if db_product:
            db_product.current_stock = qty
        else:
            new_prod = models.Product(sku=sku, name=sku, current_stock=qty, safety_stock=0)
            db.add(new_prod)
    db.commit()

    records = stock_df.where(pd.notnull(stock_df), None).to_dict(orient="records")
    return {"message": "Stock processed and DB updated", "errors": in_errs + out_errs + [s_err] if s_err else [], "data": records}

@router.get("/stock", response_model=List[schemas.Product])
def get_current_stock(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    """Fetch current stock levels from the database. Sales/Manager can view this."""
    return db.query(models.Product).all()
