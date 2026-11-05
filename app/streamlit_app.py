import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Prédiction du churn", page_icon="📉", layout="wide")

# ---------- Chargement du modèle ----------
ROOT = Path(__file__).resolve().parent.parent

@st.cache_resource
def load_model():
    saved = joblib.load(ROOT / "models" / "model.pkl")
    return saved["pipeline"], saved["threshold"]

pipe, threshold = load_model()
prep = pipe.named_steps["prep"]
model = pipe.named_steps["model"]

# ---------- Explication SHAP (compatible avec tous les modèles) ----------
def explain(client_df):
    X_t = prep.transform(client_df)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    names = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]

    if type(model).__name__ == "XGBClassifier":
        import xgboost as xgb
        contribs = model.get_booster().predict(xgb.DMatrix(X_t), pred_contribs=True)[0, :-1]
    elif type(model).__name__ == "LogisticRegression":
        contribs = (X_t * model.coef_)[0]
    else:  # RandomForest
        import shap
        contribs = shap.TreeExplainer(model)(X_t).values[0, :, 1]

    return pd.Series(contribs, index=names).sort_values(key=abs, ascending=False)

# ---------- Interface ----------
st.title("📉 Prédiction du churn client")
st.write("Renseignez le profil d'un client pour estimer son risque de résiliation et comprendre pourquoi.")

with st.sidebar:
    st.header("Profil du client")
    gender = st.selectbox("Genre", ["Male", "Female"])
    senior = st.selectbox("Senior (65+)", ["Non", "Oui"])
    partner = st.selectbox("Partenaire", ["Yes", "No"])
    dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
    tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    st.subheader("Services")
    phone = st.selectbox("Téléphone", ["Yes", "No"])
    multiple = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
    opts = ["No", "Yes", "No internet service"]
    security = st.selectbox("Sécurité en ligne", opts)
    backup = st.selectbox("Sauvegarde en ligne", opts)
    protection = st.selectbox("Protection appareil", opts)
    support = st.selectbox("Support technique", opts)
    tv = st.selectbox("Streaming TV", opts)
    movies = st.selectbox("Streaming films", opts)

    st.subheader("Contrat & paiement")
    contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Facture dématérialisée", ["Yes", "No"])
    payment = st.selectbox("Paiement", ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Charges mensuelles ($)", 18.0, 120.0, 70.0)

client = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": 1 if senior == "Oui" else 0,
    "Partner": partner, "Dependents": dependents, "tenure": tenure,
    "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
    "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
    "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
}])

# ---------- Prédiction ----------
proba = pipe.predict_proba(client)[0, 1]
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Probabilité de churn", f"{proba:.0%}")
    if proba >= threshold:
        st.error(f"⚠️ Client à risque (seuil = {threshold:.0%})")
        st.write("**Action suggérée :** contacter le client avec une offre de rétention.")
    else:
        st.success(f"✅ Client fidèle (seuil = {threshold:.0%})")
    st.progress(float(proba))

with col2:
    st.subheader("Pourquoi cette prédiction ?")
    top = explain(client).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top.index, top.values, color=["#d62728" if v > 0 else "#2ca02c" for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("← réduit le risque   |   augmente le risque →")
    st.pyplot(fig)

with st.expander("Voir les données du client"):
    st.dataframe(client)
    