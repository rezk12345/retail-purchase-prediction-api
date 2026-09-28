# Retail Purchase Prediction API

A FastAPI service that predicts whether a customer will make a purchase after a retail promotion. It uses a machine learning model trained on retail sales data and saves every prediction to a PostgreSQL database.

Built during my Data Science internship at Smart Vision.

## Features

- Machine learning model: soft-voting ensemble of Logistic Regression and Random Forest, trained with SMOTE to handle class imbalance (ROC-AUC: 0.85)
- REST API built with FastAPI
- User registration and OAuth2 login with JWT tokens and hashed passwords (passlib)
- Tokens expire after 30 minutes
- PostgreSQL database using SQLAlchemy
- Paginated list of saved predictions
- Verification script confirming the API matches the trained model's outputs
- Database password and secret key stored in a `.env` file, not in the code

## Tech Stack

Python, FastAPI, scikit-learn, imbalanced-learn, Pandas, SQLAlchemy, PostgreSQL, Pydantic, PyJWT, passlib, python-dotenv

## Project Structure

```
├── app/
│   ├── main.py            # API endpoints
│   ├── auth.py            # Password hashing, JWT tokens and login
│   ├── database.py        # Database connection
│   ├── dependencies.py    # Database session (get_db)
│   ├── models.py          # Database tables
│   ├── schemas.py         # Request and response formats
│   └── ml/
│       ├── predictor.py   # Loads the model and prepares input
│       └── *.joblib       # Saved model files
├── train_and_save.py      # Trains the model and saves it
├── verify_predictor.py    # Checks the API matches the model
├── Alpha Group Project (Ensemble Model).ipynb   # Data analysis and model development
├── .env.example           # Template for environment variables
└── requirements.txt
```

## How to Run

1. Create and activate a virtual environment:
```
   python -m venv venv
   venv\Scripts\activate
```
2. Install the requirements:
```
   pip install -r requirements.txt
```
3. Create a PostgreSQL database called `sv_project`.
4. Copy `.env.example` to a new file called `.env` and fill in your values:
```
   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost/sv_project
   SECRET_KEY=your-secret-key
```
5. Train the model (only needed once):
```
   python train_and_save.py
```
6. Start the API:
```
   uvicorn app.main:app --reload
```
7. Open http://127.0.0.1:8000/docs

## How to Use

1. Create a user with **POST /register**.
2. Click **Authorize** at the top of the docs page and log in with your username and password.
3. Make predictions with **POST /predictions**.

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /register | Create a user |
| POST | /token | Log in and receive a JWT token |
| POST | /predictions | Make a prediction (login required) |
| GET | /predictions?page=1&size=10 | List predictions with pagination (login required) |
| GET | /predictions/{id} | Get one prediction (login required) |
| DELETE | /predictions/{id} | Delete a prediction (login required) |

## Example Request

```json
{
  "country": "Germany",
  "productgroup": "HARDWARE ACCESSORIES",
  "category": "RUNNING",
  "style": "slim",
  "sizes": "xxs,xs,s,m,l,xl,xxl",
  "gender": "women",
  "sales": 2,
  "regular_price": 6.95,
  "current_price": 4.95,
  "cost": 1.29,
  "promo1": 0,
  "promo2": 0,
  "retailweek": "2017-02-26",
  "rgb_r_main_col": 181,
  "rgb_g_main_col": 181,
  "rgb_b_main_col": 181,
  "rgb_r_sec_col": 205,
  "rgb_g_sec_col": 155,
  "rgb_b_sec_col": 155
}
```

## Author

Mohamed Rezk: [GitHub](https://github.com/rezk12345)