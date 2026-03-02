from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import math

from .. import schemas, models
from .auth import get_current_user

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.data_processor import get_anchor_customers, predict_reorder

router = APIRouter()

class ForecastRequest(BaseModel):
    sales_data: List[Dict[str, Any]]
    prediction_window_days: int = 90

def clean_dict_nan(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        elif isinstance(v, dict):
            clean[k] = clean_dict_nan(v)
        elif isinstance(v, list):
            clean[k] = [clean_dict_nan(i) if isinstance(i, dict) else (None if isinstance(i, float) and math.isnan(i) else i) for i in v]
        else:
            clean[k] = v
    return clean

@router.post("/run")
async def run_forecast(
    request: ForecastRequest,
    current_user: models.User = Depends(get_current_user)
):
    """
    Runs the custom deterministic statistical model for demand forecasting.
    Takes structured JSON sales data (parsed from Excel earlier) and returns predictions.
    """
    if not request.sales_data:
        raise HTTPException(status_code=400, detail="No sales data provided for prediction")
        
    df = pd.DataFrame(request.sales_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    anchors_df, a_errs = get_anchor_customers(df)
    if anchors_df.empty:
        raise HTTPException(status_code=400, detail="Could not identify anchor customers from the provided data.")
        
    forecast_df, p_errs = predict_reorder(df, anchors_df, future_days=request.prediction_window_days)
    
    # NaN to None for JSON
    forecast_records = [clean_dict_nan(rd) for rd in forecast_df.to_dict(orient="records")]
    anchor_records = [clean_dict_nan(rd) for rd in anchors_df.to_dict(orient="records")]
    
    return {
        "anchor_customers": anchor_records,
        "forecast": forecast_records,
        "errors": a_errs + p_errs
    }
