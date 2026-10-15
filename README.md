# Retail Purchase Prediction API

A FastAPI service that predicts whether a customer will make a purchase after a retail promotion. It uses a machine learning model trained on retail sales data and saves every prediction to a PostgreSQL database.

Built during my Data Science internship at Smart Vision.

## Features

- Streamlit web interface: login, prediction form, paginated history, view and delete
- Centralised state management in the frontend
- Gemini-powered plain-language explanation of each prediction
- Machine learning model: soft-voting ensemble of Logistic Regression and Random Forest, trained with SMOTE to handle class imbalance (ROC-AUC: 0.85)
- REST API built with FastAPI
- User registration and OAuth2 login with JWT tokens and hashed passwords (passlib)
- Tokens expire after 30 minutes
- PostgreSQL database using SQLAlchemy
- Paginated list of saved predictions
- Verification script confirming the API matches the trained model's outputs
- Database password and secret key stored in a `.env` file, not in the code

## Tech Stack

Python, FastAPI, scikit-learn, imbalanced-learn, Pandas, SQLAlchemy, PostgreSQL, Pydantic, PyJWT, passlib, python-dotenv, Streamlit, Requests

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
├── streamlit_app.py        # Streamlit frontend with state management
└── app/gemini_service.py   # Gemini integration (retry + fallback models)      # Streamlit frontend (with centralised state management)
└── requirements.txt

```

Create a Gemini API key in Google AI Studio and add to .env:
   GEMINI_API_KEY=your-key
   GEMINI_MODEL=gemini-3.5-flash-lite
   GEMINI_FALLBACK_MODELS=gemini-3.1-flash-lite,gemini-flash-lite-latest
Model names change over time; list the ones your key can use in AI Studio.

## How to Run


1. Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Create a .env file (copy .env.example) and fill in:
   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost/sv_project
   SECRET_KEY=your-secret-key

4. Make sure PostgreSQL is running and the sv_project database exists.

5. (Optional) Retrain the model:
   python train_and_save.py

6. Start the API (terminal 1):
   uvicorn app.main:app --reload

7. Start the frontend (terminal 2, venv activated):
   streamlit run streamlit_app.py

8. Open http://localhost:8501
   (API docs: http://127.0.0.1:8000/docs)


## State Management
Streamlit re-runs the script on every interaction, so remembered values live in
st.session_state, managed in one section of streamlit_app.py: DEFAULT_STATE,
init_state(), login_user() / logout_user(), and callbacks. A live "Session state"
panel in the sidebar shows the state in real time. The backend is stateless
(JWT), and permanent data is stored in PostgreSQL.

## Gemini Integration
POST /predictions/{id}/explain sends the saved prediction to Gemini and returns a
plain-language explanation. The API key stays on the server. Requests are
retried, then fall back to backup models if a model is overloaded (503), retired
(404) or rate-limited (429). The explanation is an interpretation, not the model's
exact reasoning.

## How to Use

1. Open the Streamlit app and register an account.
2. Log in.
3. Fill in the form on "New prediction" and click Predict.
4. Review past predictions on the "History" page.

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