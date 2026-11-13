"""Preprocesamiento compartido entre el entrenamiento y la app de Streamlit.

Replica EXACTAMENTE las transformaciones del notebook (secciones 4 y 5), de modo
que un cliente nuevo se transforme igual que los datos con los que se entrenó.
"""
import numpy as np
import pandas as pd

COLS_NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
COLS_INTERNET = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
COLS_BINARIAS = ["Partner", "Dependents", "PhoneService", "MultipleLines",
                 "PaperlessBilling"] + COLS_INTERNET

# Orden exacto de columnas de X (idéntico al notebook)
COLUMNAS_X = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "MonthlyCharges", "TotalCharges", "InternetService_Fiber optic",
    "InternetService_No", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


def limpiar_crudo(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza de las secciones 4.1 y 4.3 sobre datos 'crudos' (texto)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df[COLS_INTERNET] = df[COLS_INTERNET].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def a_features(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """Convierte datos crudos (formato original del CSV) en la matriz X SIN escalar."""
    df = limpiar_crudo(df_crudo)
    X = pd.DataFrame(index=df.index)
    X["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    X["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    for col in COLS_BINARIAS:
        X[col] = df[col].map({"No": 0, "Yes": 1})
    X["tenure"] = df["tenure"].astype(float)
    X["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    X["TotalCharges"] = df["TotalCharges"].astype(float)
    # One-Hot con drop_first=True (categorías base: DSL, Month-to-month, Bank transfer)
    X["InternetService_Fiber optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    X["InternetService_No"] = (df["InternetService"] == "No").astype(int)
    X["Contract_One year"] = (df["Contract"] == "One year").astype(int)
    X["Contract_Two year"] = (df["Contract"] == "Two year").astype(int)
    X["PaymentMethod_Credit card (automatic)"] = (
        df["PaymentMethod"] == "Credit card (automatic)").astype(int)
    X["PaymentMethod_Electronic check"] = (
        df["PaymentMethod"] == "Electronic check").astype(int)
    X["PaymentMethod_Mailed check"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    return X[COLUMNAS_X]


def escalar(X: pd.DataFrame, scaler) -> pd.DataFrame:
    """Escala solo las 3 numéricas continuas con el scaler ajustado en entrenamiento."""
    X = X.copy()
    X[COLS_NUM] = scaler.transform(X[COLS_NUM])
    return X
