from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from preprocesamiento import a_features, escalar

BASE = Path(__file__).parent
ROJO, VERDE, AZUL, NARANJA = "#D1495B", "#4C9F70", "#00798C", "#EDAE49"

st.set_page_config(page_title="Permanencia de clientes · Telco", page_icon="📞", layout="wide")


@st.cache_resource
def cargar_modelos():
    return joblib.load(BASE / "modelos" / "modelos_churn.joblib")


@st.cache_resource
def cargar_eval():
    return joblib.load(BASE / "modelos" / "evaluacion.joblib")


@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / "data" / "telco.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Abandono"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


B, EV, DF = cargar_modelos(), cargar_eval(), cargar_datos()
MODELOS = B["top"]
TASA_BASE = DF["Abandono"].mean()

TRAD = {"Yes": "Sí", "No": "No", "Female": "Mujer", "Male": "Hombre",
        "Month-to-month": "Mes a mes", "One year": "1 año", "Two year": "2 años",
        "DSL": "DSL", "Fiber optic": "Fibra óptica",
        "Electronic check": "Cheque electrónico", "Mailed check": "Cheque por correo",
        "Bank transfer (automatic)": "Transferencia bancaria (automática)",
        "Credit card (automatic)": "Tarjeta de crédito (automática)"}
t = lambda v: TRAD.get(v, v)

NOMBRE_VAR = {
    "gender": "Género", "SeniorCitizen": "Adulto mayor", "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes", "tenure": "Antigüedad (meses)",
    "PhoneService": "Servicio telefónico", "MultipleLines": "Varias líneas",
    "OnlineSecurity": "Seguridad en línea", "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de dispositivo", "TechSupport": "Soporte técnico",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming películas",
    "PaperlessBilling": "Factura electrónica", "MonthlyCharges": "Cargo mensual",
    "TotalCharges": "Cargo total", "InternetService_Fiber optic": "Internet: fibra óptica",
    "InternetService_No": "Sin internet", "Contract_One year": "Contrato 1 año",
    "Contract_Two year": "Contrato 2 años",
    "PaymentMethod_Credit card (automatic)": "Pago: tarjeta automática",
    "PaymentMethod_Electronic check": "Pago: cheque electrónico",
    "PaymentMethod_Mailed check": "Pago: cheque por correo",
}



def predecir(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de ABANDONO por modelo (calibrada) + promedio de consenso."""
    X = escalar(a_features(df_crudo), B["scaler"])
    out = {}
    for nombre, info in B["modelos"].items():
        m = info["modelo"]
        s = m.predict_proba(X)[:, 1] if info["tipo"] == "clasificador" else m.predict(X)
        out[nombre] = info["calibrador"].predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
    res = pd.DataFrame(out, index=df_crudo.index)
    res["Consenso"] = res[MODELOS].mean(axis=1)
    return res


def nivel_riesgo(p: float) -> tuple[str, str]:
    if p < 0.15:
        return "Bajo", VERDE
    if p < 0.35:
        return "Medio", NARANJA
    if p < 0.60:
        return "Alto", "#E4572E"
    return "Muy alto", ROJO


def estilo_plotly(fig, alto=380):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=13), legend=dict(orientation="h", y=-0.2))
    return fig



with st.sidebar:
    st.title("📞 Churn Telco")
    st.caption("Predicción de la probabilidad de que un cliente **permanezca** con el servicio.")
    st.markdown("### Umbral de decisión")
    umbral = st.slider("Se marca como «se va» si su probabilidad de abandono es ≥",
                       0.05, 0.80, 0.27, 0.01,
                       help="Valores bajos detectan más abandonos (más recall) pero generan más "
                            "falsos positivos. El valor inicial equivale a la tasa de abandono "
                            "del dataset (≈ 27 %), igual que en el notebook.")
    st.markdown("### Modelos utilizados")
    for n in MODELOS:
        m = B["modelos"][n]["metricas"]
        st.markdown(f"**{n}**  \nROC-AUC {m['ROC-AUC']:.3f} · Recall {m['Recall (Churn)']:.3f} "
                    f"· F1 {m['F1 (Churn)']:.3f}")
    st.caption("Los 3 modelos con mayor ROC-AUC del notebook. Las probabilidades están "
               "calibradas con validación cruzada, por lo que son comparables entre modelos.")

st.title("Predicción de permanencia de clientes")
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Cliente individual", "📂 Carga masiva (CSV)",
                                  "📊 Análisis del dataset", "🏆 Desempeño de modelos"])


with tab1:
    st.subheader("Datos del cliente")
    si_no = ["No", "Yes"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Perfil**")
        gender = st.selectbox("Género", ["Female", "Male"], format_func=t)
        senior = st.selectbox("¿Adulto mayor (65+)?", [0, 1], format_func=lambda v: "Sí" if v else "No")
        partner = st.selectbox("¿Tiene pareja?", si_no, format_func=t)
        dependents = st.selectbox("¿Tiene dependientes?", si_no, format_func=t)
        tenure = st.slider("Antigüedad (meses)", 0, 72, 12)
        monthly = st.number_input("Cargo mensual (USD)", 15.0, 125.0, 70.0, 0.5)
        auto_total = st.checkbox("Estimar cargo total (antigüedad × cargo mensual)", True)
        total = (tenure * monthly if auto_total else
                 st.number_input("Cargo total acumulado (USD)", 0.0, 9000.0, 840.0, 10.0))
    with c2:
        st.markdown("**Servicios**")
        phone = st.selectbox("Servicio telefónico", si_no, index=1, format_func=t)
        multi = st.selectbox("Varias líneas", si_no, format_func=t, disabled=(phone == "No"))
        internet = st.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"],
                                index=1, format_func=lambda v: "Sin internet" if v == "No" else t(v))
        sin_int = internet == "No"
        security = st.selectbox("Seguridad en línea", si_no, format_func=t, disabled=sin_int)
        backup = st.selectbox("Respaldo en línea", si_no, format_func=t, disabled=sin_int)
        device = st.selectbox("Protección de dispositivo", si_no, format_func=t, disabled=sin_int)
    with c3:
        st.markdown("**Servicios (cont.) y contrato**")
        tech = st.selectbox("Soporte técnico", si_no, format_func=t, disabled=sin_int)
        tv = st.selectbox("Streaming TV", si_no, format_func=t, disabled=sin_int)
        movies = st.selectbox("Streaming películas", si_no, format_func=t, disabled=sin_int)
        contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"], format_func=t)
        paperless = st.selectbox("Factura electrónica", si_no, index=1, format_func=t)
        pago = st.selectbox("Método de pago", ["Electronic check", "Mailed check",
                                               "Bank transfer (automatic)", "Credit card (automatic)"],
                            format_func=t)

    forzar = lambda v, cond: "No" if cond else v
    cliente = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": forzar(multi, phone == "No"),
        "InternetService": internet, "OnlineSecurity": forzar(security, sin_int),
        "OnlineBackup": forzar(backup, sin_int), "DeviceProtection": forzar(device, sin_int),
        "TechSupport": forzar(tech, sin_int), "StreamingTV": forzar(tv, sin_int),
        "StreamingMovies": forzar(movies, sin_int), "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": pago,
        "MonthlyCharges": monthly, "TotalCharges": total}])

    prob = predecir(cliente).iloc[0]
    p_cons = prob["Consenso"]
    nivel, color = nivel_riesgo(p_cons)
    votos = int(sum(prob[m] >= umbral for m in MODELOS))
    se_va = p_cons >= umbral

    st.divider()
    st.subheader("Resultado")
    g, r = st.columns([1, 1.15])
    with g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=(1 - p_cons) * 100,
            number=dict(suffix=" %", font=dict(size=46)),
            title=dict(text="Probabilidad de que PERMANEZCA"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#1f2937", thickness=0.25),
                       steps=[dict(range=[0, 40], color="#F3B6BE"),
                              dict(range=[40, 65], color="#F9D9A6"),
                              dict(range=[65, 85], color="#CFE8D8"),
                              dict(range=[85, 100], color="#9ED3B4")])))
        st.plotly_chart(estilo_plotly(fig, 300), width="stretch")
    with r:
        if se_va:
            st.error(f"### ⚠️ Probable abandono\n{votos} de {len(MODELOS)} modelos lo marcan como "
                     f"«se va» con el umbral actual ({umbral:.0%}).")
        else:
            st.success(f"### ✅ Probable permanencia\n{votos} de {len(MODELOS)} modelos lo marcan como "
                       f"«se va» con el umbral actual ({umbral:.0%}).")
        m1, m2, m3 = st.columns(3)
        m1.metric("Prob. de permanecer", f"{1 - p_cons:.1%}")
        m2.metric("Prob. de abandono", f"{p_cons:.1%}",
                  delta=f"{(p_cons - TASA_BASE) * 100:+.1f} pts vs. promedio", delta_color="inverse")
        m3.metric("Nivel de riesgo", nivel)

        det = pd.DataFrame({
            "Modelo": MODELOS + ["Consenso (promedio)"],
            "Prob. permanecer": [1 - prob[m] for m in MODELOS] + [1 - p_cons],
            "Prob. abandono": [prob[m] for m in MODELOS] + [p_cons],
            "Decisión": ["Se va" if prob[m] >= umbral else "Permanece" for m in MODELOS]
                        + ["Se va" if se_va else "Permanece"]})
        st.dataframe(det.style.format({"Prob. permanecer": "{:.1%}", "Prob. abandono": "{:.1%}"}),
                     hide_index=True, width="stretch")
        spread = prob[MODELOS].max() - prob[MODELOS].min()
        if spread > 0.15:
            st.warning(f"Los modelos discrepan ({spread * 100:.0f} puntos de diferencia): "
                       "conviene tratar este caso con cautela.")

    st.subheader("¿Qué factores influyen en este cliente?")
    a, b = st.columns(2)
    with a:
        if "Regresión logística" in B["modelos"]:
            lr = B["modelos"]["Regresión logística"]["modelo"]
            Xc = escalar(a_features(cliente), B["scaler"]).iloc[0]
            medias = escalar(a_features(DF), B["scaler"]).mean()
            contrib = (lr.coef_[0] * (Xc - medias)).rename(index=NOMBRE_VAR)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])[::-1]
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker_color=[ROJO if v > 0 else VERDE for v in top.values]))
            fig.update_layout(title="Contribución al riesgo vs. un cliente promedio",
                              xaxis_title="← reduce el riesgo | aumenta el riesgo →")
            st.plotly_chart(estilo_plotly(fig, 370), width="stretch")
            st.caption("Calculado con la regresión logística (modelo interpretable): "
                       "coeficiente × diferencia respecto al cliente promedio.")
    with b:
        reglas = [
            ("Contrato mes a mes", lambda d: d["Contract"] == "Month-to-month",
             "Ofrecer descuento por migrar a contrato de 1 o 2 años."),
            ("Internet de fibra óptica", lambda d: d["InternetService"] == "Fiber optic",
             "Revisar calidad/precio del servicio de fibra y ofrecer paquetes."),
            ("Pago con cheque electrónico", lambda d: d["PaymentMethod"] == "Electronic check",
             "Incentivar el pago automático (tarjeta o transferencia)."),
            ("Antigüedad ≤ 12 meses", lambda d: d["tenure"] <= 12,
             "Programa de acompañamiento en el primer año."),
            ("Sin soporte técnico", lambda d: (d["InternetService"] != "No") & (d["TechSupport"] == "No"),
             "Ofrecer soporte técnico gratuito por un periodo de prueba."),
            ("Sin seguridad en línea", lambda d: (d["InternetService"] != "No") & (d["OnlineSecurity"] == "No"),
             "Ofrecer seguridad en línea como servicio de cortesía."),
            ("Cargo mensual ≥ 70 USD", lambda d: d["MonthlyCharges"] >= 70,
             "Revisar si un plan más ajustado le conviene."),
            ("Factura electrónica", lambda d: d["PaperlessBilling"] == "Yes", ""),
            ("Adulto mayor", lambda d: d["SeniorCitizen"] == 1,
             "Canal de atención dedicado y planes simples."),
            ("Contrato de 2 años", lambda d: d["Contract"] == "Two year", ""),
            ("Contrato de 1 año", lambda d: d["Contract"] == "One year", ""),
            ("Antigüedad > 24 meses", lambda d: d["tenure"] > 24, ""),
            ("Con soporte técnico", lambda d: (d["InternetService"] != "No") & (d["TechSupport"] == "Yes"), ""),
            ("Con seguridad en línea", lambda d: (d["InternetService"] != "No") & (d["OnlineSecurity"] == "Yes"), ""),
            ("Pago automático", lambda d: d["PaymentMethod"].str.contains("automatic"), ""),
        ]
        riesgo, protec, acciones = [], [], []
        for nombre, f, accion in reglas:
            if bool(f(cliente).iloc[0]):
                tasa = DF.loc[f(DF), "Abandono"].mean()
                linea = f"**{nombre}** — {tasa:.0%} de abandono en clientes con esta característica"
                if tasa > TASA_BASE * 1.15:
                    riesgo.append(linea)
                    if accion:
                        acciones.append(accion)
                elif tasa < TASA_BASE * 0.85:
                    protec.append(linea)
        st.markdown(f"**Factores de riesgo detectados** (promedio general: {TASA_BASE:.0%})")
        st.markdown("\n".join(f"- 🔴 {x}" for x in riesgo) if riesgo else "_Ninguno relevante._")
        st.markdown("**Factores protectores**")
        st.markdown("\n".join(f"- 🟢 {x}" for x in protec) if protec else "_Ninguno relevante._")
        if acciones and p_cons >= 0.15:
            st.markdown("**Acciones de retención sugeridas**")
            st.markdown("\n".join(f"- {x}" for x in acciones[:4]))


with tab2:
    st.subheader("Predicción para muchos clientes")
    st.write("Sube un CSV con el **mismo formato del dataset original** (columnas: gender, "
             "SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, "
             "InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, "
             "StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, "
             "MonthlyCharges, TotalCharges). `customerID` y `Churn` son opcionales; si "
             "incluyes `Churn`, se calculan además las métricas reales.")
    col_a, col_b = st.columns([2, 1])
    archivo = col_a.file_uploader("Archivo CSV", type="csv")
    ejemplo = DF.drop(columns="Abandono").sample(200, random_state=7)
    col_b.download_button("⬇️ Descargar CSV de ejemplo", ejemplo.to_csv(index=False).encode(),
                          "ejemplo_clientes.csv", "text/csv")
    usar_demo = col_b.checkbox("Probar con 200 clientes de ejemplo")

    datos = None
    if archivo is not None:
        datos = pd.read_csv(archivo)
    elif usar_demo:
        datos = ejemplo.copy()

    if datos is not None:
        requeridas = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                      "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                      "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                      "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
        faltan = [c for c in requeridas if c not in datos.columns]
        if faltan:
            st.error(f"Faltan columnas en el archivo: {', '.join(faltan)}")
        else:
            try:
                res = predecir(datos)
            except Exception as e:  # valores no esperados en las categorías
                st.error(f"No se pudo procesar el archivo. Revisa que las categorías coincidan "
                         f"con las del dataset original. Detalle: {e}")
                st.stop()
            if res.isnull().values.any():
                st.error("Hay valores vacíos o categorías no reconocidas en el archivo.")
                st.stop()

            salida = datos.copy()
            for m in MODELOS:
                salida[f"Prob. permanecer · {m}"] = (1 - res[m]).round(4)
            salida["Prob. permanecer (consenso)"] = (1 - res["Consenso"]).round(4)
            salida["Prob. abandono (consenso)"] = res["Consenso"].round(4)
            salida["Nivel de riesgo"] = res["Consenso"].map(lambda p: nivel_riesgo(p)[0])
            salida["Predicción"] = np.where(res["Consenso"] >= umbral, "Se va", "Permanece")

            n = len(salida)
            n_va = int((res["Consenso"] >= umbral).sum())
            ingreso_riesgo = datos.loc[res["Consenso"] >= umbral, "MonthlyCharges"].sum()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Clientes analizados", f"{n:,}")
            k2.metric("Clasificados «se va»", f"{n_va:,}", f"{n_va / n:.1%} del total", delta_color="off")
            k3.metric("Abandonos esperados (Σ prob.)", f"{res['Consenso'].sum():.0f}")
            k4.metric("Ingreso mensual en riesgo", f"${ingreso_riesgo:,.0f}")

            orden = ["Bajo", "Medio", "Alto", "Muy alto"]
            colores = {"Bajo": VERDE, "Medio": NARANJA, "Alto": "#E4572E", "Muy alto": ROJO}
            g1, g2 = st.columns(2)
            with g1:
                fig = px.histogram(res, x="Consenso", nbins=30, color_discrete_sequence=[AZUL],
                                   labels={"Consenso": "Probabilidad de abandono"},
                                   title="Distribución de la probabilidad de abandono")
                fig.add_vline(x=umbral, line_dash="dash", line_color=ROJO,
                              annotation_text="umbral", annotation_position="top")
                st.plotly_chart(estilo_plotly(fig), width="stretch")
            with g2:
                seg = salida["Nivel de riesgo"].value_counts().reindex(orden).fillna(0).reset_index()
                seg.columns = ["Nivel", "Clientes"]
                fig = px.bar(seg, x="Nivel", y="Clientes", color="Nivel", color_discrete_map=colores,
                             title="Clientes por nivel de riesgo", text="Clientes")
                fig.update_layout(showlegend=False)
                st.plotly_chart(estilo_plotly(fig), width="stretch")

            g3, g4 = st.columns(2)
            with g3:
                riesgo_contrato = (pd.DataFrame({"Contrato": datos["Contract"].map(t),
                                                 "Prob. abandono": res["Consenso"]})
                                   .groupby("Contrato", as_index=False).mean())
                fig = px.bar(riesgo_contrato, x="Contrato", y="Prob. abandono", text_auto=".1%",
                             color_discrete_sequence=[ROJO], title="Riesgo promedio por contrato")
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(estilo_plotly(fig), width="stretch")
            with g4:
                riesgo_int = (pd.DataFrame({"Internet": datos["InternetService"].map(
                    lambda v: "Sin internet" if v == "No" else t(v)), "Prob. abandono": res["Consenso"]})
                              .groupby("Internet", as_index=False).mean())
                fig = px.bar(riesgo_int, x="Internet", y="Prob. abandono", text_auto=".1%",
                             color_discrete_sequence=[NARANJA], title="Riesgo promedio por tipo de internet")
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(estilo_plotly(fig), width="stretch")

            if "Churn" in datos.columns:
                st.markdown("#### Desempeño real sobre tu archivo")
                y_real = datos["Churn"].map({"No": 0, "Yes": 1})
                pred = (res["Consenso"] >= umbral).astype(int)
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("ROC-AUC", f"{roc_auc_score(y_real, res['Consenso']):.3f}")
                q2.metric("Recall (Churn)", f"{recall_score(y_real, pred):.3f}")
                q3.metric("Precisión (Churn)", f"{precision_score(y_real, pred, zero_division=0):.3f}")
                q4.metric("F1 (Churn)", f"{f1_score(y_real, pred):.3f}")
                st.caption("Nota: si el archivo incluye clientes del dataset original, estos "
                           "estuvieron en el entrenamiento y las métricas serán optimistas.")

            st.markdown("#### Resultados por cliente")
            solo_riesgo = st.checkbox("Mostrar solo clientes clasificados «se va»")
            vista = salida[salida["Predicción"] == "Se va"] if solo_riesgo else salida
            vista = vista.sort_values("Prob. abandono (consenso)", ascending=False)
            st.dataframe(vista, hide_index=True, width="stretch")
            st.download_button("⬇️ Descargar resultados (CSV)", salida.to_csv(index=False).encode(),
                               "predicciones_churn.csv", "text/csv")


with tab3:
    st.subheader("Panorama de los clientes")
    perdidos = DF.loc[DF["Abandono"] == 1, "MonthlyCharges"].sum()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clientes", f"{len(DF):,}")
    k2.metric("Tasa de abandono", f"{TASA_BASE:.1%}")
    k3.metric("Antigüedad media", f"{DF['tenure'].mean():.0f} meses")
    k4.metric("Cargo mensual medio", f"${DF['MonthlyCharges'].mean():.0f}")
    k5.metric("Ingreso mensual perdido", f"${perdidos:,.0f}",
              f"{perdidos / DF['MonthlyCharges'].sum():.0%} del total", delta_color="off")

    nombres_cat = {"Contract": "Contrato", "InternetService": "Servicio de internet",
                   "PaymentMethod": "Método de pago", "TechSupport": "Soporte técnico",
                   "OnlineSecurity": "Seguridad en línea", "PaperlessBilling": "Factura electrónica",
                   "SeniorCitizen": "Adulto mayor", "Partner": "Pareja", "Dependents": "Dependientes",
                   "gender": "Género", "PhoneService": "Servicio telefónico",
                   "MultipleLines": "Varias líneas", "StreamingTV": "Streaming TV",
                   "StreamingMovies": "Streaming películas", "OnlineBackup": "Respaldo en línea",
                   "DeviceProtection": "Protección de dispositivo"}
    a, b = st.columns(2)
    with a:
        var = st.selectbox("Abandono según…", list(nombres_cat), format_func=nombres_cat.get)
        tmp = DF.assign(**{var: DF[var].map(lambda v: {0: "No", 1: "Sí"}.get(v, t(v)) if not isinstance(v, str)
                                            else t(v))})
        g = tmp.groupby(var)["Abandono"].agg(["mean", "size"]).reset_index()
        g.columns = [nombres_cat[var], "Tasa de abandono", "Clientes"]
        fig = px.bar(g, x=nombres_cat[var], y="Tasa de abandono", text_auto=".1%", hover_data=["Clientes"],
                     color="Tasa de abandono", color_continuous_scale=["#9ED3B4", "#F9D9A6", ROJO],
                     title=f"Tasa de abandono por {nombres_cat[var].lower()}")
        fig.add_hline(y=TASA_BASE, line_dash="dash", line_color="gray", annotation_text="promedio")
        fig.update_yaxes(tickformat=".0%")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(estilo_plotly(fig), width="stretch")
    with b:
        num = st.selectbox("Distribución de…", ["tenure", "MonthlyCharges", "TotalCharges"],
                           format_func=lambda v: {"tenure": "Antigüedad (meses)",
                                                  "MonthlyCharges": "Cargo mensual",
                                                  "TotalCharges": "Cargo total"}[v])
        fig = px.histogram(DF.assign(Cliente=DF["Churn"].map({"No": "Permanece", "Yes": "Se va"})),
                           x=num, color="Cliente", barmode="overlay", opacity=0.65, nbins=40,
                           color_discrete_map={"Permanece": VERDE, "Se va": ROJO},
                           title="Clientes que permanecen vs. que se van")
        st.plotly_chart(estilo_plotly(fig), width="stretch")

    c, d = st.columns(2)
    with c:
        bins = pd.cut(DF["tenure"], [-1, 6, 12, 24, 36, 48, 60, 72],
                      labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
        g = DF.groupby(bins, observed=True)["Abandono"].mean().reset_index()
        fig = px.line(g, x="tenure", y="Abandono", markers=True, color_discrete_sequence=[ROJO],
                      labels={"tenure": "Antigüedad (meses)", "Abandono": "Tasa de abandono"},
                      title="El abandono cae con la antigüedad")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(estilo_plotly(fig), width="stretch")
    with d:
        piv = (DF.pivot_table(index="Contract", columns="InternetService", values="Abandono", aggfunc="mean")
               .rename(index=t, columns=lambda v: "Sin internet" if v == "No" else t(v)))
        fig = px.imshow(piv, text_auto=".0%", color_continuous_scale=["#CFE8D8", "#F9D9A6", ROJO],
                        title="Abandono: contrato × tipo de internet", aspect="auto")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(estilo_plotly(fig), width="stretch")

    e, f = st.columns(2)
    with e:
        Xd = a_features(DF)
        cor = Xd.corrwith(DF["Abandono"]).rename(index=NOMBRE_VAR).sort_values()
        cor = pd.concat([cor.head(6), cor.tail(6)])
        fig = go.Figure(go.Bar(x=cor.values, y=cor.index, orientation="h",
                               marker_color=[ROJO if v > 0 else VERDE for v in cor.values]))
        fig.update_layout(title="Correlación de cada variable con el abandono (extremos)")
        st.plotly_chart(estilo_plotly(fig, 420), width="stretch")
    with f:
        DF["Segmento"] = pd.cut(DF["MonthlyCharges"], [0, 35, 70, 90, 200],
                                labels=["< $35", "$35–70", "$70–90", "> $90"])
        g = DF.groupby("Segmento", observed=True).agg(Abandono=("Abandono", "mean"),
                                                     Clientes=("Abandono", "size")).reset_index()
        fig = px.bar(g, x="Segmento", y="Abandono", text_auto=".1%", hover_data=["Clientes"],
                     color_discrete_sequence=[AZUL], labels={"Abandono": "Tasa de abandono",
                                                             "Segmento": "Cargo mensual"},
                     title="Abandono por rango de cargo mensual")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(estilo_plotly(fig, 420), width="stretch")

    st.markdown("#### Hallazgos clave")
    r_mes = DF.loc[DF["Contract"] == "Month-to-month", "Abandono"].mean()
    r_dos = DF.loc[DF["Contract"] == "Two year", "Abandono"].mean()
    r_fib = DF.loc[DF["InternetService"] == "Fiber optic", "Abandono"].mean()
    r_new = DF.loc[DF["tenure"] <= 12, "Abandono"].mean()
    r_chk = DF.loc[DF["PaymentMethod"] == "Electronic check", "Abandono"].mean()
    st.markdown(
        f"- Los contratos **mes a mes** abandonan **{r_mes:.0%}**, frente a **{r_dos:.0%}** en contratos de 2 años.\n"
        f"- Los clientes con **12 meses o menos** de antigüedad abandonan **{r_new:.0%}**.\n"
        f"- La **fibra óptica** tiene una tasa de abandono de **{r_fib:.0%}**, la más alta entre los tipos de internet.\n"
        f"- Pagar con **cheque electrónico** se asocia con **{r_chk:.0%}** de abandono.")


with tab4:
    st.subheader("Los 3 mejores modelos sobre el conjunto de prueba (1.409 clientes)")
    y_test = EV["y_test"]
    probs = {n: EV["test"][n]["prob"] for n in MODELOS}
    probs["Consenso"] = np.mean([probs[n] for n in MODELOS], axis=0)

    cuadro = EV["cuadro"].set_index("Modelo").loc[MODELOS, ["Tipo", "MAE", "RMSE", "R²",
                                                          "Recall (Churn)", "F1 (Churn)", "ROC-AUC"]]
    st.dataframe(cuadro.style.format({c: "{:.4f}" for c in cuadro.columns[1:]})
                 .highlight_min(subset=["MAE", "RMSE"], color="#9ED3B4")
                 .highlight_max(subset=["R²", "Recall (Churn)", "F1 (Churn)", "ROC-AUC"], color="#9ED3B4"),
                 width="stretch")
    st.caption("Métricas del notebook (umbral original de cada modelo). MAE y RMSE: menor es mejor; "
               "el resto: mayor es mejor. Los tres modelos quedan prácticamente empatados en ROC-AUC.")

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for n, col in zip(list(probs), [AZUL, NARANJA, VERDE, ROJO]):
            fpr, tpr, _ = roc_curve(y_test, probs[n])
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{n} (AUC {roc_auc_score(y_test, probs[n]):.3f})",
                            line=dict(color=col, width=3 if n == "Consenso" else 2,
                                      dash="dot" if n == "Consenso" else "solid"))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color="gray", dash="dash"))
        fig.update_layout(title="Curvas ROC", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Recall", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(estilo_plotly(fig, 420), width="stretch")
    with b:
        ts = np.linspace(0.05, 0.90, 86)
        pc = probs["Consenso"]
        curva = pd.DataFrame({"Umbral": ts,
                              "Precisión": [precision_score(y_test, pc >= u, zero_division=0) for u in ts],
                              "Recall": [recall_score(y_test, pc >= u) for u in ts],
                              "F1": [f1_score(y_test, pc >= u) for u in ts]})
        fig = px.line(curva, x="Umbral", y=["Precisión", "Recall", "F1"], title="Efecto del umbral (consenso)",
                      color_discrete_sequence=[AZUL, ROJO, NARANJA])
        fig.add_vline(x=umbral, line_dash="dash", line_color="gray", annotation_text="umbral actual")
        fig.update_layout(yaxis_title="Valor", legend_title="")
        st.plotly_chart(estilo_plotly(fig, 420), width="stretch")
        mejor = curva.loc[curva["F1"].idxmax()]
        st.caption(f"El mejor F1 del consenso se logra con un umbral de ≈ {mejor['Umbral']:.2f} "
                   f"(F1 = {mejor['F1']:.3f}). Con el umbral actual: "
                   f"Recall {recall_score(y_test, pc >= umbral):.3f}, "
                   f"Precisión {precision_score(y_test, pc >= umbral, zero_division=0):.3f}.")

    st.markdown("#### Matrices de confusión con el umbral actual")
    cols = st.columns(4)
    for col, n in zip(cols, list(probs)):
        cm = confusion_matrix(y_test, probs[n] >= umbral)
        fig = px.imshow(cm, text_auto=True, x=["Permanece", "Se va"], y=["Permanece", "Se va"],
                        color_continuous_scale="Blues", labels=dict(x="Predicho", y="Real"))
        fig.update_coloraxes(showscale=False)
        fig.update_layout(title=dict(text=n, font=dict(size=13)))
        col.plotly_chart(estilo_plotly(fig, 300), width="stretch")

    c, d = st.columns(2)
    with c:
        fig = px.histogram(pd.DataFrame({"p": pc, "Real": np.where(y_test == 1, "Se va", "Permanece")}),
                           x="p", color="Real", barmode="overlay", opacity=0.65, nbins=30,
                           color_discrete_map={"Permanece": VERDE, "Se va": ROJO},
                           labels={"p": "Probabilidad de abandono (consenso)"},
                           title="Separación entre clientes que se van y que permanecen")
        fig.add_vline(x=umbral, line_dash="dash", line_color="gray")
        st.plotly_chart(estilo_plotly(fig, 380), width="stretch")
    with d:
        dec = pd.qcut(pc, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"Predicha": pc, "Real": y_test, "dec": dec}).groupby("dec").mean()
        fig = go.Figure()
        fig.add_scatter(x=cal["Predicha"], y=cal["Real"], mode="lines+markers", name="Modelo (consenso)",
                        line=dict(color=AZUL))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                        line=dict(color="gray", dash="dash"))
        fig.update_layout(title="Calibración: ¿la probabilidad refleja la frecuencia real?",
                          xaxis_title="Probabilidad predicha", yaxis_title="Tasa real de abandono")
        st.plotly_chart(estilo_plotly(fig, 380), width="stretch")

    st.markdown("#### Ganancia por segmentos (¿a quién contactar primero?)")
    orden = np.argsort(-pc)
    y_ord = np.asarray(y_test)[orden]
    gan = pd.DataFrame({"% de clientes contactados": np.arange(1, len(y_ord) + 1) / len(y_ord) * 100,
                        "% de abandonos capturados": np.cumsum(y_ord) / y_ord.sum() * 100})
    fig = px.line(gan, x="% de clientes contactados", y="% de abandonos capturados",
                  color_discrete_sequence=[ROJO], title="Curva de ganancia acumulada")
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Sin modelo (al azar)",
                    line=dict(color="gray", dash="dash"))
    top20 = gan.iloc[int(len(gan) * 0.2) - 1]["% de abandonos capturados"]
    st.plotly_chart(estilo_plotly(fig, 380), width="stretch")
    st.caption(f"Contactando al 20 % de clientes con mayor riesgo se captura ≈ {top20:.0f} % de los abandonos reales.")

    st.markdown("#### Importancia de las variables")
    sel = st.selectbox("Modelo", list(EV["importancias"]))
    imp = EV["importancias"][sel]
    s = imp["valores"].rename(index=NOMBRE_VAR)
    s = s.reindex(s.abs().sort_values(ascending=False).index[:12])[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[(ROJO if v > 0 else VERDE) if "signo" in imp["tipo"] else AZUL
                                         for v in s.values]))
    fig.update_layout(title=f"{sel}: {imp['tipo']}")
    st.plotly_chart(estilo_plotly(fig, 420), width="stretch")
    if "signo" in imp["tipo"]:
        st.caption("Rojo = aumenta la probabilidad de abandono; verde = la reduce (variables numéricas estandarizadas).")

st.divider()
st.caption("Modelo de apoyo a la decisión: las probabilidades describen patrones históricos del dataset "
           "y no sustituyen el criterio del equipo comercial.")
