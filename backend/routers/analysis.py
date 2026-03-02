from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from .. import schemas, models, database
from .auth import get_current_user

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=schemas.HistoricalAnalysis)
def save_analysis(
    analysis: schemas.HistoricalAnalysisCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_analysis = models.HistoricalAnalysis(
        user_id=current_user.id,
        date_run=datetime.utcnow(),
        forecast_parameters=analysis.forecast_parameters,
        result_json=analysis.result_json
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

@router.get("/", response_model=list[schemas.HistoricalAnalysis])
def get_user_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    history = db.query(models.HistoricalAnalysis).filter(models.HistoricalAnalysis.user_id == current_user.id).order_by(models.HistoricalAnalysis.date_run.desc()).all()
    return history
