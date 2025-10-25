# Julio Borrayo Garza 737656

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Título y descripción
st.title("🔮 Predicción con Regresión Lineal Simple")
st.write("Aplicación interactiva para entrenar un modelo de regresión lineal y visualizar las predicciones.")
st.write("Selecciona la variable dependiente (Y) y la variable independiente (X).")

# Cargar datos
st.subheader("1️⃣ Cargar datos")
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if uploaded_file is not None:
    # Leer datos
    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())

    # Seleccionar columnas
    columnas = data.columns.tolist()
    x_col = st.selectbox("Selecciona la variable independiente (X)", columnas)
    y_col = st.selectbox("Selecciona la variable dependiente (Y)", columnas)

    # Definir variables
    X = data[[x_col]].values
    y = data[y_col].values

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X, y)

    # Mostrar ecuación del modelo
    st.subheader("2️⃣ Ecuación del modelo")
    pendiente = model.coef_[0]
    intercepto = model.intercept_
    st.latex(f"{y_col} = {pendiente:.4f} \\times {x_col} + {intercepto:.4f}")

    # Calcular R²
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    st.metric("Coeficiente de determinación (R²)", f"{r2:.4f}")

    # Predicción interactiva
    st.subheader("3️⃣ Predicción con valores nuevos")
    valor_x = st.number_input(f"Ingrese un valor para {x_col}:", value=float(X.mean()))
    prediccion = model.predict(np.array([[valor_x]]))[0]
    st.write(f"🔮 Predicción de {y_col}: **{prediccion:.4f}**")

    # Visualización
    st.subheader("4️⃣ Visualización de la regresión")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", label="Datos reales")
    ax.plot(X, y_pred, color="red", linewidth=2, label="Línea de regresión")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("👆 Sube un archivo CSV para continuar.")
