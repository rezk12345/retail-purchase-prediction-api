"""
Streamlit frontend for the SV Project - Purchase Prediction API.

Run (with the FastAPI server already running on port 8000):
    streamlit run streamlit_app.py
"""
import copy
import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = 15  # seconds

st.set_page_config(page_title="Purchase Prediction", page_icon="🛒", layout="wide")


# ------------------------------------------------------------------ helpers
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """'#b5b5b5' -> (181, 181, 181). The API expects separate R, G, B ints."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# =====================================================================
# STATE MANAGEMENT
# Streamlit re-runs this whole file on every click, so normal variables are
# forgotten. Everything the app must REMEMBER lives in st.session_state.
# We keep it in ONE place: defaults, getters, and the only functions that
# are allowed to change it (login_user, logout_user, reset_page ...).
# =====================================================================
DEFAULT_STATE = {
    # --- authentication state ---
    "token": None,            # the JWT; None means "not logged in"
    "username": None,
    # --- navigation / UI state ---
    "nav": "New prediction",  # which page is open
    "page": 1,                # current history page
    "page_size": 10,          # rows per history page
    "confirm_delete": None,   # id waiting for delete confirmation
    "flash": None,            # one-time message (e.g. session expired)
    # --- data state ---
    "last_result": None,      # last prediction shown on the form page
    "predictions_made": 0,    # counter for this login session
    "explanations": {},       # {prediction_id: Gemini explanation text}
}
# Cleared on login/logout. ("nav" is NOT listed: it belongs to the sidebar radio
# widget, and Streamlit forbids changing a widget's key after it is drawn.)
SESSION_ONLY = ["page", "page_size", "confirm_delete",
                "last_result", "predictions_made", "explanations"]


def init_state():
    """Give every key a default value the first time the app runs."""
    for key, value in DEFAULT_STATE.items():
        st.session_state.setdefault(key, copy.deepcopy(value))


def is_logged_in() -> bool:
    return st.session_state.token is not None


def _clear_session_data():
    for key in SESSION_ONLY:
        st.session_state[key] = copy.deepcopy(DEFAULT_STATE[key])


def login_user(token: str, username: str):
    """The ONLY place that marks the user as logged in."""
    _clear_session_data()
    st.session_state.token = token
    st.session_state.username = username


def logout_user(message: str | None = None):
    """The ONLY place that logs out: wipes auth + session data."""
    st.session_state.token = None
    st.session_state.username = None
    _clear_session_data()
    st.session_state.flash = message
    st.rerun()


# --- callbacks: functions Streamlit runs BEFORE the page re-runs ---
def reset_page():
    st.session_state.page = 1


def change_page_size():
    st.session_state.page_size = st.session_state.size_widget
    reset_page()  # a new page size makes the old page number meaningless


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def render_state_panel():
    """Shows the current state live, so you can SEE state management working."""
    with st.sidebar.expander("🔍 Session state (live)"):
        view = {k: st.session_state.get(k) for k in DEFAULT_STATE}
        if view["token"]:
            view["token"] = "..." + view["token"][-8:]  # never show the full token
        st.json(view)


def api(method: str, path: str, timeout: int = TIMEOUT, **kwargs):
    """Call the API. Returns (response, error_message)."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is uvicorn running?"
    except requests.exceptions.Timeout:
        return None, "The API took too long to respond."

    # Token expired (30 min) or invalid -> send the user back to login
    if resp.status_code == 401 and st.session_state.get("token"):
        logout_user("Your session expired. Please log in again.")

    return resp, None


def error_text(resp) -> str:
    """Pull a readable message out of a FastAPI error response."""
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text
    if isinstance(detail, list):  # 422 validation errors
        return "; ".join(f"{'.'.join(map(str, d['loc'][1:]))}: {d['msg']}" for d in detail)
    return str(detail)


# ------------------------------------------------------------------ login / register
def auth_page():
    st.title("🛒 Purchase Prediction")
    st.caption("Log in to predict whether a customer will purchase after a promotion.")

    if st.session_state.flash:
        st.warning(st.session_state.flash)
        st.session_state.flash = None

    _, center, _ = st.columns([1, 2, 1])
    with center:
        tab_login, tab_register = st.tabs(["Log in", "Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", width="stretch")
            if submitted:
                if not username or not password:
                    st.error("Enter both username and password.")
                else:
                    # /token expects FORM data (not JSON) -> use data=, not json=
                    resp, err = api("POST", "/token", data={"username": username, "password": password})
                    if err:
                        st.error(err)
                    elif resp.status_code == 200:
                        login_user(resp.json()["access_token"], username)
                        st.rerun()
                    else:
                        st.error(error_text(resp))

        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("Choose a username")
                new_pass = st.text_input("Choose a password", type="password")
                confirm = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", width="stretch")
            if submitted:
                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                elif new_pass != confirm:
                    st.error("Passwords do not match.")
                else:
                    resp, err = api("POST", "/register", json={"username": new_user, "password": new_pass})
                    if err:
                        st.error(err)
                    elif resp.status_code == 200:
                        st.success("Account created. Switch to the Log in tab.")
                    else:
                        st.error(error_text(resp))


# ------------------------------------------------------------------ predict page
def explain_section(prediction_id: int):
    """Button + result for the Gemini explanation (kept in session state)."""
    st.markdown("**Explain this result (Gemini)**")
    cache = st.session_state.explanations

    if st.button("✨ Explain with Gemini", key=f"explain_{prediction_id}"):
        with st.spinner("Asking Gemini..."):
            resp, err = api("POST", f"/predictions/{prediction_id}/explain",
                            timeout=60, headers=auth_headers())
        if err:
            st.error(err)
        elif resp.status_code == 200:
            cache[prediction_id] = resp.json()["explanation"]  # remembered in state
        else:
            st.error(error_text(resp))

    if prediction_id in cache:
        st.info(cache[prediction_id])
        st.caption("AI-generated interpretation. It is not the model's exact reasoning.")


def show_result(rec: dict):
    purchase = rec["prediction"] == 1
    prob = rec["probability"]
    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", rec["result"])
    c2.metric("Purchase probability", f"{prob:.1%}")
    c3.metric("Saved as ID", rec["id"])
    st.progress(min(max(prob, 0.0), 1.0))
    if purchase:
        st.success("The model expects this customer to **purchase**.")
    else:
        st.info("The model expects **no purchase**.")
    explain_section(rec["id"])


def predict_page():
    st.header("New prediction")

    with st.form("predict_form"):
        st.markdown("**Product**")
        c1, c2, c3 = st.columns(3)
        country = c1.selectbox("Country", ["Austria", "France", "Germany"])
        productgroup = c2.selectbox("Product group", ["HARDWARE ACCESSORIES", "SHOES", "SHORTS", "SWEATSHIRTS"])
        category = c3.selectbox(
            "Category", ["FOOTBALL GENERIC", "GOLF", "INDOOR", "RELAX CASUAL", "RUNNING", "TRAINING"]
        )
        c1, c2, c3 = st.columns(3)
        style = c1.selectbox("Style", ["regular", "slim", "wide"])
        sizes = c2.selectbox("Sizes", ["xs,s,m,l,xl", "xxs,xs,s,m,l,xl,xxl"])
        gender = c3.selectbox("Gender", ["kids", "men", "unisex", "women"])

        st.markdown("**Pricing and sales**")
        c1, c2, c3, c4 = st.columns(4)
        regular_price = c1.number_input("Regular price", min_value=0.01, value=6.95, step=0.5, format="%.2f")
        current_price = c2.number_input("Current price", min_value=0.0, value=4.95, step=0.5, format="%.2f")
        cost = c3.number_input("Cost", min_value=0.0, value=1.29, step=0.5, format="%.2f")
        sales = c4.number_input("Sales", min_value=0.0, value=2.0, step=1.0)

        st.markdown("**Promotion and timing**")
        c1, c2, c3 = st.columns(3)
        promo1 = c1.checkbox("Promo 1 active")
        promo2 = c2.checkbox("Promo 2 active")
        retailweek = c3.date_input("Retail week", value=date(2017, 2, 26))

        st.markdown("**Colors**")
        c1, c2 = st.columns(2)
        main_hex = c1.color_picker("Main color", "#b5b5b5")
        sec_hex = c2.color_picker("Secondary color", "#cd9b9b")

        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        # Guard against the division in predictor.py (current_price / regular_price)
        if regular_price <= 0:
            st.error("Regular price must be greater than 0.")
            return

        mr, mg, mb = hex_to_rgb(main_hex)
        sr, sg, sb = hex_to_rgb(sec_hex)
        payload = {
            "country": country,
            "productgroup": productgroup,
            "category": category,
            "style": style,
            "sizes": sizes,
            "gender": gender,
            "sales": sales,
            "regular_price": regular_price,
            "current_price": current_price,
            "cost": cost,
            "promo1": int(promo1),
            "promo2": int(promo2),
            "retailweek": retailweek.isoformat(),
            "rgb_r_main_col": mr,
            "rgb_g_main_col": mg,
            "rgb_b_main_col": mb,
            "rgb_r_sec_col": sr,
            "rgb_g_sec_col": sg,
            "rgb_b_sec_col": sb,
        }
        with st.spinner("Running the model..."):
            resp, err = api("POST", "/predictions", json=payload, headers=auth_headers())
        if err:
            st.error(err)
        elif resp.status_code == 200:
            st.session_state.last_result = resp.json()
            st.session_state.predictions_made += 1
        else:
            st.error(error_text(resp))

    if st.session_state.last_result:
        st.divider()
        show_result(st.session_state.last_result)


# ------------------------------------------------------------------ history page
def history_page():
    st.header("Prediction history")

    sizes = [5, 10, 20, 50]
    st.selectbox("Rows per page", sizes, index=sizes.index(st.session_state.page_size),
                 key="size_widget", on_change=change_page_size)
    size = st.session_state.page_size
    page = st.session_state.page

    resp, err = api("GET", "/predictions", params={"page": page, "size": size}, headers=auth_headers())
    if err:
        st.error(err)
        return
    if resp.status_code != 200:
        st.error(error_text(resp))
        return

    data = resp.json()
    total, items = data["total"], data["items"]
    pages = max(1, -(-total // size))  # ceiling division

    # If rows were deleted and the current page no longer exists, step back
    if page > pages:
        st.session_state.page = pages
        st.rerun()

    st.caption(f"{total} prediction(s) saved · page {page} of {pages}")

    if items:
        df = pd.DataFrame(items)
        df["probability"] = (df["probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            df[["id", "country", "productgroup", "category", "retailweek", "result", "probability", "created_at"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No predictions yet. Make one on the New prediction page.")

    # Pagination controls
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("⬅ Previous", disabled=page <= 1, width="stretch"):
        st.session_state.page -= 1
        st.rerun()
    c2.markdown(f"<div style='text-align:center;padding-top:6px'>Page {page} / {pages}</div>", unsafe_allow_html=True)
    if c3.button("Next ➡", disabled=page >= pages, width="stretch"):
        st.session_state.page += 1
        st.rerun()

    st.divider()
    st.subheader("Look up or delete by ID")
    c1, c2, c3 = st.columns([2, 1, 1])
    pred_id = c1.number_input("Prediction ID", min_value=1, step=1, value=items[0]["id"] if items else 1)

    if c2.button("View", width="stretch"):
        r, e = api("GET", f"/predictions/{int(pred_id)}", headers=auth_headers())
        if e:
            st.error(e)
        elif r.status_code == 200:
            st.json(r.json())
        else:
            st.error(error_text(r))

    if c3.button("Delete", type="primary", width="stretch"):
        st.session_state.confirm_delete = int(pred_id)

    if st.session_state.get("confirm_delete"):
        target = st.session_state.confirm_delete
        st.warning(f"Delete prediction {target}? This cannot be undone.")
        y, n, _ = st.columns([1, 1, 4])
        if y.button("Yes, delete"):
            r, e = api("DELETE", f"/predictions/{target}", headers=auth_headers())
            st.session_state.confirm_delete = None
            if e:
                st.error(e)
            elif r.status_code == 200:
                st.success(r.json()["message"])
                st.rerun()
            else:
                st.error(error_text(r))
        if n.button("Cancel"):
            st.session_state.confirm_delete = None
            st.rerun()


# ------------------------------------------------------------------ main router
def main():
    init_state()
    if not is_logged_in():
        auth_page()
        return

    # 1) Sidebar part that decides which page to show (must be drawn first)
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        choice = st.radio("Navigate", ["New prediction", "History"],
                          key="nav", label_visibility="collapsed")

    # 2) Run the page. This is where the state CHANGES (e.g. a new prediction).
    if choice == "New prediction":
        predict_page()
    else:
        history_page()

    # 3) Draw everything that DISPLAYS state AFTER the page has run, so it
    #    always shows the latest values (drawing it earlier showed stale data).
    with st.sidebar:
        st.metric("Predictions this session", st.session_state.predictions_made)
        st.divider()
        if st.button("Log out", width="stretch"):
            logout_user()
        st.caption(f"API: {API_URL}")
    render_state_panel()


main()