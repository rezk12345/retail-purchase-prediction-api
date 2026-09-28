from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app import models, schemas
from app.auth import hash_password, verify_password, create_access_token, get_current_user
from app.database import Base, engine
from app.dependencies import get_db
from app.ml.predictor import predict

Base.metadata.create_all(bind=engine)

app = FastAPI(title="SV Project - Purchase Prediction API")


@app.get("/")
def root():
    return {"message": "Purchase Prediction API is running. Go to /docs to try it."}


@app.post("/register", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")

    new_user = models.User(username=user.username, hashed_password=hash_password(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/token", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return {"access_token": create_access_token(user.username), "token_type": "bearer"}


@app.post("/predictions", response_model=schemas.PredictionOutput)
def create_prediction(
    payload: schemas.PredictionInput,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    prediction, probability = predict(payload.model_dump())

    record = models.Prediction(
        country=payload.country,
        productgroup=payload.productgroup,
        category=payload.category,
        retailweek=payload.retailweek,
        input_data=payload.model_dump(mode="json"),
        prediction=prediction,
        probability=round(probability, 4),
        result="Purchase" if prediction == 1 else "No Purchase",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/predictions", response_model=schemas.PaginatedPredictions)
def list_predictions(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    total = db.query(models.Prediction).count()
    items = (
        db.query(models.Prediction)
        .order_by(models.Prediction.id.desc())
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )
    return {"total": total, "page": page, "size": size, "items": items}


@app.get("/predictions/{prediction_id}", response_model=schemas.PredictionOutput)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    record = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record


@app.delete("/predictions/{prediction_id}")
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    record = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(record)
    db.commit()
    return {"message": f"Prediction {prediction_id} deleted"}