from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import math

from .. import schemas, models, database
from .auth import get_current_user

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_processor import get_anchor_customers, predict_reorder, compute_material_requirements

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ForecastRequest(BaseModel):
    sales_data: List[Dict[str, Any]]
    prediction_window_days: int = 90

def clean_dict_nan(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        elif pd.isna(v):
            clean[k] = None
        elif isinstance(v, dict):
            clean[k] = clean_dict_nan(v)
        elif isinstance(v, list):
            clean[k] = [clean_dict_nan(i) if isinstance(i, dict) else (None if isinstance(i, float) and math.isnan(i) else (None if pd.isna(i) else i)) for i in v]
        else:
            clean[k] = v
    return clean

@router.post("/run")
async def run_forecast(
    request: ForecastRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Runs the custom deterministic statistical model for demand forecasting.
    Takes structured JSON sales data (parsed from Excel earlier) and returns predictions.
    Now automatically queries the BOM database and combines it for material requirements.
    """
    if not request.sales_data:
        raise HTTPException(status_code=400, detail="No sales data provided for prediction")
        
    df = pd.DataFrame(request.sales_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    anchors_df, a_errs = get_anchor_customers(df)
    if anchors_df.empty:
        raise HTTPException(status_code=400, detail="Could not identify anchor customers from the provided data.")
        
    forecast_df, p_errs = predict_reorder(df, anchors_df)
    
    # -------------------------------------------------------------
    # NEW: AUTOMATIC BOM EXPLOSION IN FORECAST ROUTE
    # -------------------------------------------------------------
    boms = db.query(models.BOMRecord).all()
    bom_data = [
        {
            "product": b.product,
            "copper_type": b.copper_type,
            "copper_weight_kg": b.copper_weight_kg,
            "lamination_type": b.lamination_type,
            "lamination_weight_kg": b.lamination_weight_kg,
            "bobbin_type": b.bobbin_type,
            "other_reqs": b.other_reqs
        } for b in boms
    ]
    bom_df = pd.DataFrame(bom_data) if bom_data else pd.DataFrame()
    
    products = db.query(models.Product).all()
    stock_df = pd.DataFrame([{"Material": p.sku, "Current Stock": p.current_stock} for p in products]) if products else pd.DataFrame()
    
    req_df, req_errs = compute_material_requirements(forecast_df, bom_df, stock_df)
    
    # NaN to None for JSON
    forecast_records = [clean_dict_nan(rd) for rd in forecast_df.to_dict(orient="records")]
    anchor_records = [clean_dict_nan(rd) for rd in anchors_df.to_dict(orient="records")]
    req_records = [clean_dict_nan(rd) for rd in req_df.to_dict(orient="records")] if not req_df.empty else []
    
    errs = (a_errs if a_errs else []) + (p_errs if p_errs else []) + ([req_errs] if req_errs else [])
    
    return {
        "anchor_customers": anchor_records,
        "forecast": forecast_records,
        "material_requirements": req_records,
        "errors": errs
    }
