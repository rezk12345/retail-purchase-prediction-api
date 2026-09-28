from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, JSON
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    country = Column(String)
    productgroup = Column(String)
    category = Column(String)
    retailweek = Column(Date)
    input_data = Column(JSON)
    prediction = Column(Integer)
    probability = Column(Float)
    result = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)