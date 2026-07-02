import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Deposit Predictor", page_icon="🏦", layout="centered")

# ── Load trained model ──
data = joblib.load("bank_model.pkl")
model = data["model"]
train_columns = data["columns"]
cat_cols = data["cat_cols"]
num_cols = data["num_cols"]
cat_options = data["cat_options"]

st.title("🏦 Bank Deposit Predictor")
st.write("Enter the customer's details, and the model (Random Forest) will predict whether they will subscribe to a term deposit.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", cat_options["job"])
    marital = st.selectbox("Marital Status", cat_options["marital"])
    education = st.selectbox("Education", cat_options["education"])
    default = st.selectbox("Has credit in default?", cat_options["default"])
    balance = st.number_input("Balance", value=1000, step=100)
    housing = st.selectbox("Has housing loan?", cat_options["housing"])
    loan = st.selectbox("Has personal loan?", cat_options["loan"])

with col2:
    contact = st.selectbox("Contact Type", cat_options["contact"])
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Last Contact Month", cat_options["month"])
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=200)
    campaign = st.number_input("Number of Contacts This Campaign", min_value=1, value=2)
    pdays = st.number_input("Days Since Previous Contact (-1 = never contacted)", value=-1)
    previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
    poutcome = st.selectbox("Previous Campaign Outcome", cat_options["poutcome"])

st.divider()

if st.button("🔮 Predict", use_container_width=True):
    # Build a single-row DataFrame matching training input format
    input_dict = {
        "job": job, "marital": marital, "education": education, "default": default,
        "housing": housing, "loan": loan, "contact": contact, "month": month,
        "poutcome": poutcome,
        "age": age, "balance": balance, "day": day, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous
    }
    input_df = pd.DataFrame([input_dict])

    # Same encoding used in training (get_dummies + align columns)
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.success("✅ The customer is predicted to SUBSCRIBE (Deposit = Yes)")
    else:
        st.error("❌ The customer is predicted to DECLINE (Deposit = No)")

    st.metric("Probability of Yes", f"{probability*100:.1f}%")
    st.progress(float(probability))
