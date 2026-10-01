"""
Streamlit frontend for the SV Project - Purchase Prediction API.

Run (with the FastAPI server already running on port 8000):
    streamlit run streamlit_app.py
"""
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


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def logout(message: str | None = None):
    st.session_state.token = None
    st.session_state.username = None
    if message:
        st.session_state.flash = message
    st.rerun()


def api(method: str, path: str, **kwargs):
    """Call the API. Returns (response, error_message)."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is uvicorn running?"
    except requests.exceptions.Timeout:
        return None, "The API took too long to respond."

    # Token expired (30 min) or invalid -> send the user back to login
    if resp.status_code == 401 and st.session_state.get("token"):
        logout("Your session expired. Please log in again.")

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


# ------------------------------------------------------------------ session state
st.session_state.setdefault("token", None)
st.session_state.setdefault("username", None)
st.session_state.setdefault("page", 1)
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("flash", None)


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
                        st.session_state.token = resp.json()["access_token"]
                        st.session_state.username = username
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
        else:
            st.error(error_text(resp))

    if st.session_state.last_result:
        st.divider()
        show_result(st.session_state.last_result)


# ------------------------------------------------------------------ history page
def history_page():
    st.header("Prediction history")

    size = st.selectbox("Rows per page", [5, 10, 20, 50], index=1)
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
    if not st.session_state.token:
        auth_page()
        return

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        choice = st.radio("Navigate", ["New prediction", "History"], label_visibility="collapsed")
        st.divider()
        if st.button("Log out", width="stretch"):
            logout()
        st.caption(f"API: {API_URL}")

    if choice == "New prediction":
        predict_page()
    else:
        history_page()


main()
