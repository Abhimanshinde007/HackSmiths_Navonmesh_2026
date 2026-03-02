from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import math

from .. import schemas, models, database
from .auth import get_current_user

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_processor import ingest_bom_excel, compute_material_requirements

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
class BomExplodeRequest(BaseModel):
    bom_data: List[Dict[str, Any]]  # Or rely on database BOM
    forecast_data: List[Dict[str, Any]]

def clean_dict_nan(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        elif pd.isna(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean

@router.post("/upload")
async def upload_bom_excel(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ingest BOM Excel directly and store it into the PostgreSQL database.
    This replaces the in-memory pandas BOM with persistent BillOfMaterials table records.
    """
    content = await file.read()
    file_obj = io.BytesIO(content)
    file_obj.name = file.filename
    
    df, errs = ingest_bom_excel(file_obj)
    if errs and df.empty:
        raise HTTPException(status_code=400, detail="|".join(errs))
        
    records = [clean_dict_nan(rd) for rd in df.to_dict(orient="records")]
    
    # Insert straight into models.BOMRecord
    for _, row in df.iterrows():
        prod = str(row.get('product', '')).strip()
        if not prod: continue
        
        db_bom = db.query(models.BOMRecord).filter(models.BOMRecord.product == prod).first()
        if not db_bom:
            db_bom = models.BOMRecord(product=prod)
            db.add(db_bom)
            
        db_bom.copper_type = str(row.get('copper_type', ''))
        db_bom.copper_weight_kg = float(row.get('copper_weight_kg', 0) or 0)
        db_bom.lamination_type = str(row.get('lamination_type', ''))
        db_bom.lamination_weight_kg = float(row.get('lamination_weight_kg', 0) or 0)
        db_bom.bobbin_type = str(row.get('bobbin_type', ''))
        db_bom.other_reqs = str(row.get('other_reqs', ''))
        
    db.commit()
    
    return {"message": "BOM extracted and saved to Database", "errors": errs, "data": records}

@router.post("/explode")
async def run_bom_explosion(
    request: BomExplodeRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Integrate the BOM Explosion logic to calculate raw material requirements.
    Takes the structured BOM array and the Forecast array, returns aggregated materials.
    """
    if not request.bom_data or not request.forecast_data:
        raise HTTPException(status_code=400, detail="BOM data and Forecast data are required.")
        
    bom_df = pd.DataFrame(request.bom_data)
    forecast_df = pd.DataFrame(request.forecast_data)
    
    # Needs stock df from db
    products = db.query(models.Product).all()
    stock_df = pd.DataFrame([{"Material": p.sku, "Current Stock": p.current_stock} for p in products])
    
    req_df, err = compute_material_requirements(forecast_df, bom_df, stock_df)
    if err:
        raise HTTPException(status_code=400, detail=err)
        
    results = [clean_dict_nan(rd) for rd in req_df.to_dict(orient="records")]
    return {
        "material_requirements": results
    }
