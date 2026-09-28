from datetime import date, datetime
from pydantic import BaseModel, ConfigDict


class PredictionInput(BaseModel):
    country: str
    productgroup: str
    category: str
    style: str
    sizes: str
    gender: str
    sales: float
    regular_price: float
    current_price: float
    cost: float
    promo1: int
    promo2: int
    retailweek: date
    rgb_r_main_col: int
    rgb_g_main_col: int
    rgb_b_main_col: int
    rgb_r_sec_col: int
    rgb_g_sec_col: int
    rgb_b_sec_col: int

class UserCreate(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str

class PredictionOutput(BaseModel):
    id: int
    country: str
    productgroup: str
    category: str
    retailweek: date
    prediction: int
    probability: float
    result: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
    

class PaginatedPredictions(BaseModel):
    total: int
    page: int
    size: int
    items: list[PredictionOutput]