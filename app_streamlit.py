# Importar librerías 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql.functions import mean, col, count, month, year, sum, hour, when
from src.lectura_gestiones import crear_spark, cargar_gestiones
from modelos.modelo_nlp import entrenar_modelo_nlp, limpiar_texto_udf
from modelos.pedidos.entrenar_pedidos import entrenar_modelos_pedidos
from modelos.pedidos.predictor_pedidos import predecir_pedido
from wordcloud import WordCloud, STOPWORDS
from datetime import date, timedelta

# -------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------
st.set_page_config(
    page_title="Clasificación de gestiones e impacto en colocación de pedidos",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados (opcional, puedes ajustarlo)
st.markdown("""
    <style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# FUNCIONES AUXILIARES
# -------------------------------
def grafica_conteo_porcentaje(df_spark, columna, titulo, key=None):
    """
    Genera una gráfica de barras con:
    - Eje Y: total de registros
    - Etiqueta: porcentaje (%)
    Y muestra también la tabla con totales y porcentajes.

    key: identificador único para evitar conflictos de Streamlit.
    """
    total_registros = df_spark.count()

    df_group = (
        df_spark
        .groupBy(columna)
        .agg(count("*").alias("total"))
        .withColumn("porcentaje", (col("total") / total_registros * 100))
        .orderBy(col("total").desc())
    )

    pdf = df_group.toPandas()
    pdf["porcentaje_label"] = pdf["porcentaje"].round(2).astype(str) + "%"

    fig = px.bar(
        pdf,
        x=columna,
        y="total",
        text="porcentaje_label",
        title=titulo,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="Total de registros",
        xaxis_title=columna,
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig, use_container_width=True, key=key or f"plot_{columna}_{titulo}")
    st.dataframe(pdf)

def generar_nube_palabras(df_spark, etiqueta, max_words=100):
    # Usa la columna limpia si existe
    col_texto = "nota_clean" if "nota_clean" in df_spark.columns else "nota"

    notas_pd = (
        df_spark
        .filter(col("resultado_asesor") == etiqueta)
        .select(col_texto)
        .toPandas()
    )

    if notas_pd.empty:
        return None

    texto = " ".join(notas_pd[col_texto].astype(str).tolist())

    stopwords = set(STOPWORDS)
    stopwords.update({
        "cliente", "llamada", "whatsapp", "asesor", "gestion", "gestiones",
        "producto", "productos", "servicio", "servicios", "oferta",
        "paquete", "informacion", "información", "mas", "más",
        "dijo", "comentó", "comento", "menciono", "mencionó",
        "indico", "indicó", "pidio", "pidió", "mostro", "mostró",
        "que", "de", "la", "el", "los", "las", "en", "por", "para",
        "interes", "interesado", "interesada", "interesados",
        "interesarse", "interesar",
    })

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=max_words,
        stopwords=stopwords
    ).generate(texto)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig

# -------------------------------
# CARGAR DATOS
# -------------------------------
RUTA_CSV = "datos/gestiones.csv"

@st.cache_resource
def get_spark():
    return crear_spark()

@st.cache_resource
def get_data():
    spark = get_spark()
    df = cargar_gestiones(spark, RUTA_CSV)
    # Crear columna de nota limpia (para modelo y nubes de palabras)
    if "nota_clean" not in df.columns:
        df = df.withColumn("nota_clean", limpiar_texto_udf(col("nota")))
    return df

df = get_data()

# -------------------------------
# SIDEBAR (RADIO DE PÁGINAS)
# -------------------------------
st.sidebar.info("""
**📚 Sistema de análisis y predicción de gestiones para mejorar la colocación de pedidos en categorías de conectividad, hogar y movilidad**

Universidad Anáhuac Puebla  
📚 Ciencia de Datos            

Alumna: Karla Beatriz Hernández Castro 
                
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Selecciona una sección")
page = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "🏠 Inicio",
        "📈 Análisis Exploratorio de Gestiones y Pedidos",
        "🤖 Modelo: Clasificación de Notas de Gestiones",
        "📊 Modelo: Predicción de Horario de Gestiones para Mayor Efectividad de Pedidos",
        "📝 Conclusiones",
    ],
    label_visibility="collapsed"
)

# ===============================
# PÁGINA: 🏠 INICIO
# ===============================
if page == "🏠 Inicio":
    st.markdown(
        """
        <h2 class="sub-header">
            🤖📈 Sistema de Análisis y Predicción de Gestiones para Mejorar la venta 
            de pedidos en diferentes categorías.🔮
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h3>Descripción de proyecto</h3>', unsafe_allow_html=True)

    st.markdown("""
    El sistema analiza todas las gestiones realizadas por asesores durante 2025 para comprender
    cómo responden los clientes y qué tan probable es que finalicen en una compra.

    Además, incluye un modelo que recomienda **el mejor horario para contactar al cliente**,
    basado en patrones de comportamiento y la información registrada en la nota del asesor,
    buscando aumentar la efectividad de cada gestión.
    """)

    # Propósito, reglas y problema a resolver
    st.markdown("""
    ---
    ### 🎯 Propósito del sistema: 

    Este proyecto nace para responder tres preguntas: 
    
    1. **¿Cómo es el desempeño comercial de los asesores actualmente?**
                
    2. **¿Las notas del asesor reflejan si el cliente está interesado o no en realizar un pedido?**  

    3. **¿En qué horario es mejor realizar una gestión para que el cliente compre?**  
                
    ---

    ### 🧩 Datos a analizar

    - Se cuenta con un dataset en formato CSV que contiene el resultado de la extracción y transformación de los datos de gestiones 
      realizadas en el año 2025. 
                
      El origen de este archivo es la unión de gestiones vs pedidos generados en un lapso de 24 horas por lo que se asume que se realizó 
      la compra por este motivo. 
    """)
    #SCHEMA
    st.subheader("🔎 Tipos de datos")
    st.caption(
        "Información por columna de lo que se espera recibir en cada columna del archivo para realizar el análisis"
    )
    columnas_info = [
        {
            "columna": "id_gestion",
            "tipo_dato": "int",
            "descripcion": "Identificador único de cada gestión (registro en la base).",
            "valores_esperados": "Enteros consecutivos, sin duplicados."
        },
        {
            "columna": "id_cliente",
            "tipo_dato": "string",
            "descripcion": "Identificador del cliente al que se le realizó la gestión.",
            "valores_esperados": "Formato 'C#########', puede repetirse (un cliente con varias gestiones)."
        },
        {
            "columna": "id_asesor",
            "tipo_dato": "string",
            "descripcion": "Identificador del asesor que realizó la gestión.",
            "valores_esperados": "Formato 'A#########', un asesor puede aparecer en muchas gestiones."
        },
        {
            "columna": "medio",
            "tipo_dato": "string (categórica)",
            "descripcion": "Canal por el cual se realizó la gestión.",
            "valores_esperados": "'llamada' o 'whatsapp'."
        },
        {
            "columna": "resultado_asesor",
            "tipo_dato": "string (categórica)",
            "descripcion": "Evaluación del asesor sobre el interés del cliente.",
            "valores_esperados": "'interesado' o 'no_interesado'."
        },
        {
            "columna": "fecha_hora",
            "tipo_dato": "timestamp",
            "descripcion": "Fecha y hora en la que se realizó la gestión.",
            "valores_esperados": "Timestamps entre 2025-01-01 y 2025-12-31 aprox."
        },
        {
            "columna": "nota",
            "tipo_dato": "string (texto libre)",
            "descripcion": "Comentario del asesor sobre la gestión; base para el modelo NLP.",
            "valores_esperados": "Frases en español que describen la respuesta del cliente y contexto."
        },
        {
            "columna": "pedido_generado",
            "tipo_dato": "int (0/1)",
            "descripcion": "Indica si la gestión terminó en un pedido.",
            "valores_esperados": "0 = no generó pedido, 1 = sí generó pedido."
        },
        {
            "columna": "id_pedido",
            "tipo_dato": "string",
            "descripcion": "Identificador del pedido generado a partir de la gestión (si aplica).",
            "valores_esperados": "Formato 'P#########' cuando pedido_generado=1, 'NA' cuando pedido_generado=0."
        },
        {
            "columna": "monto_pedido",
            "tipo_dato": "double",
            "descripcion": "Monto total del pedido asociado a la gestión.",
            "valores_esperados": "0 si no hubo pedido; entre ~200 y 35,000 cuando pedido_generado=1."
        },
        {
            "columna": "producto_categoria",
            "tipo_dato": "string (categórica)",
            "descripcion": "Tipo de producto asociado al pedido.",
            "valores_esperados": "'conectividad', 'movilidad', 'hogar' cuando hay pedido; 'NA' cuando no hay pedido."
        },
    ]

    tipos_df = pd.DataFrame(columnas_info)
    st.dataframe(tipos_df, use_container_width=True)

    # Revisión de nulos
    st.subheader("🧼 Limpieza de datos antes de análisis")
    st.caption(
        "Se verifica si hay valores faltantes en alguna columna que puedan afectar el entrenamiento de los modelos."
    )

    nulos_pd = df.select([
        sum(col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).toPandas().T.reset_index()

    nulos_pd.columns = ["columna", "n_nulos"]
    st.dataframe(nulos_pd)

    # Validación lógica de reglas
    st.subheader("✅ Validación de datos")
    st.caption("Se verifica la coherencia entre los pedidos y categorías para revisar que el archivo se generó correctamente")

    regla1 = df.filter((df.pedido_generado == 1) & (df.id_pedido == "NA")).count()
    regla2 = df.filter((df.pedido_generado == 0) & (df.monto_pedido != 0)).count()
    regla3 = df.filter((df.pedido_generado == 0) & (df.producto_categoria != "NA")).count()

    c1, c2, c3 = st.columns(3)
    c1.metric("Pedidos que no cuentan con número de pedio", regla1)
    c2.metric("Sin pedido pero con monto mayor a $0", regla2)
    c3.metric("Sin pedido pero con alguna categoría de hogar, movilidad o conectividad", regla3)

    # Muestra de datos
    st.subheader("🔎 Muestra de datos")
    st.caption(
        "Muestra representativa de la base para revisar visualmente los campos y el tipo de información registrada."
    )
    muestra = df.limit(1000).toPandas()
    st.dataframe(muestra)
    
    st.markdown("---")

# ===============================
# PÁGINA: 📈 Análisis Exploratorio de Gestiones y Pedidos
# ===============================
elif page == "📈 Análisis Exploratorio de Gestiones y Pedidos":
    st.subheader("📈 Análisis Exploratorio de Gestiones y Pedidos detallado")
    st.subheader("""¿Cómo es el desempeño comercial de los asesores actualmente?""")
    st.markdown(
        "En esta sección se revisará el estatus actual de las gestiones y pedidos generado de los asesores durante el 2025"
    )   
    st.markdown("---")


    st.markdown(
        "Información general de la base de datos: "
    )
    # KPIs básicos
    total_registros = df.count()
    tasa_pedidos = df.select(mean("pedido_generado").alias("tasa")).collect()[0]["tasa"]

    # Total de pedidos generados
    total_pedidos = df.filter(col("pedido_generado") == 1).count()

    # Total de pedidos NO generados
    total_no_pedidos = df.filter(col("pedido_generado") == 0).count()

    # Tasa de pedidos NO generados
    tasa_no_pedidos = total_no_pedidos / total_registros

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de gestiones realizadas en 2025", f"{total_registros:,}")
    col2.metric("Total de pedidos generados", f"{tasa_pedidos*100:.2f}%")
    col3.metric("Total de pedidos NO generados", f"{tasa_no_pedidos*100:.2f}%")

    st.info("Por 1,000,000 de gestiones solo el 40% generan un pedido")


    st.markdown("---")

    st.subheader("**📆 ¿Cómo vamos en las ventas por mes?**")

    df_mes = (
        df
        .withColumn("anio", year(col("fecha_hora")))
        .withColumn("mes", month(col("fecha_hora")))
    )

    df_gestiones = (
        df_mes.groupBy("anio", "mes")
        .count()
        .withColumnRenamed("count", "total_gestiones")
    )

    df_pedidos = (
        df_mes.groupBy("anio", "mes")
        .agg(sum(col("pedido_generado")).alias("total_pedidos"))
    )

    df_join = (
        df_gestiones.join(df_pedidos, ["anio", "mes"], "inner")
        .orderBy("anio", "mes")
    )

    pdf = df_join.toPandas()
    pdf["mes_str"] = pdf["anio"].astype(str) + "-" + pdf["mes"].astype(str).str.zfill(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pdf["mes_str"],
        y=pdf["total_gestiones"],
        name="Gestiones",
        marker=dict(color="royalblue",opacity=0.7)
    ))
    fig.add_trace(go.Scatter(
        x=pdf["mes_str"],
        y=pdf["total_pedidos"],
        name="Pedidos generados",
        mode="lines+markers",
        line=dict(width=3, color="#00FF7F")
    ))
    fig.update_layout(
        title="Gestiones totales vs pedidos generados por mes",
        xaxis_title="Mes",
        yaxis_title="Cantidad",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pdf)

    st.info("El mes de mayo tiene el mayor número de pedidos generados aunque no es el mes con mayor gestiones realizadas")

    st.subheader("📆 ¿Cómo se comporta el interés del cliente por mes?")

    # Agregar columnas de año y mes
    df_mes_int = (
        df
        .withColumn("anio", year(col("fecha_hora")))
        .withColumn("mes", month(col("fecha_hora")))
    )

    # Conteo de interesados y no interesados por mes
    df_interes = (
        df_mes_int.groupBy("anio", "mes")
        .agg(
            sum(when(col("resultado_asesor") == "interesado", 1).otherwise(0)).alias("interesado"),
            sum(when(col("resultado_asesor") == "no_interesado", 1).otherwise(0)).alias("no_interesado")
        )
        .orderBy("anio", "mes")
    )

    pdf_int = df_interes.toPandas()
    pdf_int["mes_str"] = pdf_int["anio"].astype(str) + "-" + pdf_int["mes"].astype(str).str.zfill(2)

    # --- Gráfica ---
    fig_int = go.Figure()

    # Barras – interesados
    fig_int.add_trace(go.Bar(
        x=pdf_int["mes_str"],
        y=pdf_int["interesado"],
        name="Interesado",
        marker=dict(color="royalblue", opacity=0.8)
    ))

    # Línea – no interesados
    fig_int.add_trace(go.Scatter(
        x=pdf_int["mes_str"],
        y=pdf_int["no_interesado"],
        name="No interesado",
        mode="lines+markers",
        line=dict(width=3, color="crimson")
    ))

    fig_int.update_layout(
        title="Interesados vs No interesados por mes",
        xaxis_title="Mes",
        yaxis_title="Cantidad",
        barmode="group",
        template="plotly_white"
    )

    st.plotly_chart(fig_int, use_container_width=True)
    st.dataframe(pdf_int)

    st.info("El mes de Octubre es cuando más gente no se ha interesado en adquirir un producto. " \
    "Adicional a que la tendencia indica que existe mayor cantidad de personas no interesadas que interesadas")

    st.subheader("📞📦 ¿Qué combinación de medio y categoría genera más pedidos?")

    # 1) Nos quedamos sólo con los registros que SÍ generaron pedido
    df_med_cat = (
        df.filter(col("pedido_generado") == 1)
        .groupBy("medio", "producto_categoria")
        .agg(
            count("*").alias("pedidos")
        )
    )

    pdf_med_cat = df_med_cat.toPandas()

    # Si hubiera nulos en producto_categoria, los marcamos como 'sin_categoria'
    pdf_med_cat["producto_categoria"] = (
        pdf_med_cat["producto_categoria"].fillna("sin_categoria")
    )

    # Podemos excluir 'sin_categoria' si no quieres verla en el gráfico
    pdf_med_cat = pdf_med_cat[pdf_med_cat["producto_categoria"] != "sin_categoria"]

    # 2) Total de pedidos por medio (para calcular % dentro de cada canal)
    pdf_med_cat["total_pedidos_medio"] = (
        pdf_med_cat.groupby("medio")["pedidos"].transform("sum")
    )

    pdf_med_cat["porc_pedidos_medio"] = (
        (pdf_med_cat["pedidos"] / pdf_med_cat["total_pedidos_medio"]) * 100
    ).round(2)

    # 3) Gráfico de barras: pedidos por medio × categoría
    fig_mc = px.bar(
        pdf_med_cat,
        x="producto_categoria",
        y="pedidos",
        color="medio",
        barmode="group",
        text=pdf_med_cat["porc_pedidos_medio"].astype(str) + "%",
        title="Pedidos generados por combinación Medio × Categoría",
        labels={
            "pedidos": "Pedidos generados",
            "producto_categoria": "Categoría de producto"
        },
    )

    fig_mc.update_traces(textposition="outside")
    fig_mc.update_layout(
        yaxis_title="Número de pedidos",
        xaxis_title="Categoría de producto",
    )

    st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown("### 📋 Resumen Medio × Categoría")
    st.dataframe(
        pdf_med_cat[[
            "medio",
            "producto_categoria",
            "pedidos",
            "total_pedidos_medio",
            "porc_pedidos_medio",
        ]],
        use_container_width=True
    )

    st.info("Es mejor realizar una llamada que un mensaje de Whatsapp para cualquier categoria")
    

    st.markdown("---")
    st.subheader("🏆 Top 10 asesores por pedidos generados")

    df_pedidos = (
        df.groupBy("id_asesor")
        .agg(sum(col("pedido_generado")).alias("total_pedidos"))
    )
    df_gestiones = (
        df.groupBy("id_asesor")
        .count()
        .withColumnRenamed("count", "total_gestiones")
    )
    df_asesores = (
        df_pedidos.join(df_gestiones, "id_asesor", "inner")
        .orderBy(col("total_pedidos").desc())
        .limit(10)
    )

    pdf_top = df_asesores.toPandas()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pdf_top["id_asesor"],
        y=pdf_top["total_pedidos"],
        name="Pedidos",
    ))
    fig.add_trace(go.Bar(
        x=pdf_top["id_asesor"],
        y=pdf_top["total_gestiones"],
        name="Gestiones",
        opacity=0.6
    ))
    fig.update_layout(
        title="Top 10 asesores por pedidos y gestiones",
        xaxis_title="ID Asesor",
        yaxis_title="Cantidad",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True, key="plot_top10_asesores")
    st.dataframe(pdf_top)


    st.subheader("🚨 Bottom 10 asesores con menos pedidos generados")

    df_pedidos = (
        df.groupBy("id_asesor")
        .agg(sum(col("pedido_generado")).alias("total_pedidos"))
    )
    df_gestiones = (
        df.groupBy("id_asesor")
        .count()
        .withColumnRenamed("count", "total_gestiones")
    )
    df_asesores_bottom = (
        df_pedidos.join(df_gestiones, "id_asesor", "inner")
        .orderBy(col("total_pedidos").asc())
        .limit(10)
    )

    pdf_bottom = df_asesores_bottom.toPandas()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pdf_bottom["id_asesor"],
        y=pdf_bottom["total_pedidos"],
        name="Pedidos",
        marker=dict(color="crimson")
    ))
    fig.add_trace(go.Bar(
        x=pdf_bottom["id_asesor"],
        y=pdf_bottom["total_gestiones"],
        name="Gestiones",
        opacity=0.6,
        marker=dict(color="gray")
    ))
    fig.update_layout(
        title="Bottom 10 asesores: menos pedidos y sus gestiones",
        xaxis_title="ID Asesor",
        yaxis_title="Cantidad",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True, key="plot_bottom10_asesores")
    st.dataframe(pdf_bottom)


# ===============================
# PÁGINA: 🤖 NLP
# ===============================
elif page == "🤖 Modelo: Clasificación de Notas de Gestiones":
    st.title("🧠 ¿Las notas del asesor reflejan si el cliente está interesado o no en realizar un pedido?")

    st.markdown("""
    Propósito de la pregunta: 
    """)

    st.markdown("""
    Conocer a partir de técnicas de **Procesamiento de Lenguaje Natural (NLP)** si la nota que escribe el asesor en cada gestión coincide 
    con el resultado que él asigna en los rubros de **"INTERESADO o NO INTERESADO"**:
    """)
                
    st.markdown("---")
    st.subheader("🚀 Entrenamiento del modelo NLP")
    st.caption("""
    🧠 Modelo NLP: Clasificación de notas de gestiones <br>
    1. Limpieza de texto: se normaliza la nota (nota_clean: minúsculas, sin acentos ni signos raros). <br>
    2. Tokenización: se separa la nota en palabras (tokens). <br>
    3. Eliminación de stopwords: e quitan palabras muy comunes que no aportan significado. <br>
    4. Vectorización TF-IDF: se convierte el texto a números con (TF-IDF mide qué tan importante es una palabra dentro de un texto, comparándola con todos los textos del dataset). <br>
    5. Entrenamiento: entrena una Regresión Logística para predecir resultado_asesor (interesado / no_interesado). <br>
    6. Evaluación: e calcula Accuracy y F1-score y se arma la matriz de confusión para los datos de prueba.<br>
    """, unsafe_allow_html=True)

    if st.button("Entrenar modelo NLP (Regresión Logística)", type="primary"):
        with st.spinner("Entrenando modelo NLP sobre notas de gestiones, por favor espera..."):
            metrics, confusion_df_spark, modelo_nlp = entrenar_modelo_nlp(df)

        labels = metrics["labels"]
        confusion_pdf = confusion_df_spark.toPandas()

        def decode_label(idx):
            try:
                return labels[int(idx)]
            except Exception:
                return str(idx)

        confusion_pdf["label_str"] = confusion_pdf["label"].apply(decode_label)
        confusion_pdf["prediction_str"] = confusion_pdf["prediction"].apply(decode_label)
        confusion_pdf = confusion_pdf[["label_str", "prediction_str", "count"]]
        confusion_pdf = confusion_pdf.rename(columns={
            "label_str": "Etiqueta real (resultado_asesor)",
            "prediction_str": "Predicción modelo",
            "count": "Cantidad"
        })

        st.session_state["modelo_nlp"] = modelo_nlp
        st.session_state["labels_nlp"] = labels
        st.session_state["metrics_nlp"] = metrics
        st.session_state["confusion_pdf_nlp"] = confusion_pdf

        st.success("✅ Entrenamiento completado correctamente.")

    if "metrics_nlp" in st.session_state:

        metrics = st.session_state["metrics_nlp"]
        confusion_pdf = st.session_state["confusion_pdf_nlp"]
        labels = st.session_state["labels_nlp"]

        acc = metrics["accuracy"]
        f1 = metrics["f1"]
        num_train = metrics["num_train"]
        num_test = metrics["num_test"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy (¿Qué tantas notas clasifica bien el modelo?)", f"{acc*100:.2f}%")
        col2.metric("F1-score (¿Qué tan equilibrada es la calidad de la clasificación?)", f"{f1:.3f}")
        col3.metric("Registros (train/test)", f"{num_train} / {num_test}")

        st.markdown("### 🔁 Matriz de confusión para datos test")
        st.dataframe(confusion_pdf)

        cm_pivot = confusion_pdf.pivot(
            index="Etiqueta real (resultado_asesor)",
            columns="Predicción modelo",
            values="Cantidad"
        ).fillna(0)

        fig_cm = px.imshow(
            cm_pivot,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(
                x="Predicción del modelo",
                y="Etiqueta real (resultado_asesor)",
                color="Cantidad"
            ),
            title="Matriz de confusión - modelo NLP (Regresión Logística)"
        )
        fig_cm.update_layout(xaxis_side="top")
        st.plotly_chart(fig_cm, use_container_width=True, key="cm_heatmap_nlp")

        st.markdown(
            "<p style='font-size: 0.85rem; color: gray;'>"
            "La matriz de confusión permite analizar dónde el modelo acierta más y en qué casos se confunde.<br>"
            "• El modelo identifica correctamente a la mayoría de los clientes no interesados (más de 108 mil casos bien clasificados).<br>"
            "• También clasifica de forma precisa a los clientes interesados, con más de 76 mil aciertos.<br>"
            "• Los errores (falsos positivos y falsos negativos) son relativamente bajos comparados con el volumen total.<br>"
            "</p>",
            unsafe_allow_html=True
        )

        # SECCIÓN DE NUBES 
        st.markdown("---")
        st.subheader("☁️ Nube de palabras: interesado vs no_interesado")

        col_int, col_no = st.columns(2)

        # 1) Nube 'interesado' (solo se genera una vez)
        if "wordcloud_interesado" not in st.session_state:
            with st.spinner("Generando nube de palabras para 'interesado'..."):
                fig_int = generar_nube_palabras(df, "interesado", max_words=80)
                st.session_state["wordcloud_interesado"] = fig_int

        with col_int:
            st.markdown("**Notas con resultado_asesor = 'interesado'**")
            if st.session_state["wordcloud_interesado"] is not None:
                st.pyplot(st.session_state["wordcloud_interesado"], use_container_width=True)
            else:
                st.info("No hay suficientes notas para 'interesado'.")

        # 2) Nube 'no_interesado' (solo se genera una vez)
        if "wordcloud_no_interesado" not in st.session_state:
            with st.spinner("Generando nube de palabras para 'no_interesado'..."):
                fig_no = generar_nube_palabras(df, "no_interesado", max_words=80)
                st.session_state["wordcloud_no_interesado"] = fig_no

        with col_no:
            st.markdown("**Notas con resultado_asesor = 'no_interesado'**")
            if st.session_state["wordcloud_no_interesado"] is not None:
                st.pyplot(st.session_state["wordcloud_no_interesado"], use_container_width=True)
            else:
                st.info("No hay suficientes notas para 'no_interesado'.")
                
        # PRUEBA EN VIVO NLP
        st.markdown("---")
        st.subheader("✍️ Prueba en vivo del modelo NLP")

        st.caption("Escribe una nota y el modelo te dirá si suena a 'interesado' o 'no_interesado'.")

        nota_input = st.text_area(
            "Escribe aquí la nota del asesor:",
            height=120,
            placeholder="Ejemplo: El cliente pidió más información sobre el paquete de internet..."
        )

        if st.button("Clasificar nota con el modelo NLP"):
            if "modelo_nlp" not in st.session_state:
                st.error("Primero entrena el modelo NLP con el botón de arriba.")
            elif not nota_input.strip():
                st.error("Por favor escribe una nota antes de clasificar.")
            else:
                spark = get_spark()

                # 1) Crear DataFrame con la nota original
                df_nota = spark.createDataFrame(
                    [(nota_input, "interesado")],  # etiqueta dummy
                    ["nota", "resultado_asesor"]
                )

                # 2) Generar la columna esperada por el pipeline
                # usar limpiar_texto_udf (tu función real)
                df_nota = df_nota.withColumn("nota_clean", limpiar_texto_udf(col("nota")))

                # 3) Recuperar modelo y labels
                modelo = st.session_state["modelo_nlp"]
                labels = st.session_state["labels_nlp"]

                # 4) Transformar y obtener predicción
                pred_row = (
                    modelo.transform(df_nota)
                        .select("prediction", "probability")
                        .collect()[0]
                )

                pred_idx = int(pred_row["prediction"])
                probs = pred_row["probability"].toArray().tolist()

                etiqueta_pred = labels[pred_idx]
                prob_pred = probs[pred_idx]

                # Mostrar resultados
                st.success(f"Predicción: **{etiqueta_pred}**")
                st.metric("Confianza del modelo", f"{prob_pred*100:.2f}%")

                df_probs = pd.DataFrame({
                    "clase": labels,
                    "probabilidad": probs
                }).sort_values("probabilidad", ascending=False)

                st.markdown("#### Detalle de probabilidades por clase")
                st.dataframe(df_probs)

    else:
        st.warning("Pulsa el botón para entrenar el modelo NLP con las notas de gestión.")



# ===============================
# PÁGINA: 📊 Modelo: Predicción de Horario de Gestiones para Mayor Efectividad de Pedidos (SOLO MODELO)
# ===============================
elif page == "📊 Modelo: Predicción de Horario de Gestiones para Mayor Efectividad de Pedidos":
    st.title("📊 ¿En qué horario es mejor realizar una gestión para que el cliente compre?")

    st.markdown("""
    Para saber el mejor horario para realizar una gestión que termine en un pedido generado es necesario considerar 
    - Medio de contacto (`medio`)
    - Resultado del asesor (`resultado_asesor`)
    - Momento de la gestión (`fecha_hora` → hora de la gestión)
    """)

    df = get_data()

    st.markdown("---")
    st.subheader("🚀 Entrenamiento del modelo para predicción de pedidos")


    st.caption("""
    📊 Modelo de predicción de pedidos: mejor horario <br>
    1. Selección de variables: medio, resultado_asesor y fecha_hora. <br>
    2. Preparación de datos: codificación y features numéricos. <br>
    3. División train/test. <br>
    4. Entrenamiento con Regresión Logística. <br>
    5. Evaluación: Accuracy, F1, Precisión, Recall, AUC y matriz de confusión. <br>
    6. Predicción en vivo. <br>
    7. Recomendación de horario simulando horas 09–21. <br>
    """, unsafe_allow_html=True)
    
    if st.button("Entrenar modelo de predicción de pedidos (Regresión Logística)", type="primary"):
        with st.spinner("Entrenando modelo de predicción de pedidos (Regresión Logística)..."):
            resultados_pedidos, pipeline_model, lr_model = entrenar_modelos_pedidos(df)

        st.session_state["pedidos_resultados"] = resultados_pedidos
        st.session_state["pedidos_pipeline"] = pipeline_model
        st.session_state["pedidos_lr_model"] = lr_model

        st.success("✅ Modelo entrenado correctamente.")

    if "pedidos_resultados" in st.session_state:
        resultados = st.session_state["pedidos_resultados"]
        res = resultados["lr"]

        st.markdown("### 📈 Resultados del modelo: **Regresión Logística**")

        c1, c2, c3= st.columns(3)
        c1.metric("Accuracy (¿Qué tantas notas clasifica bien el modelo?)", f"{res['accuracy']*100:.2f}%")
        c2.metric("F1-score (¿Qué tan equilibrada es la calidad de la clasificación?)", f"{res['f1']:.3f}")
        c3.metric("Registros train/test", f"{res['num_train']} / {res['num_test']}")

        st.markdown("#### 🔁 Matriz de confusión para datos test")
        st.dataframe(res["confusion"])

        # res["confusion"] puede ser Spark DF o pandas DF
        conf = res["confusion"]
        try:
            pdf_conf = conf.toPandas()
        except AttributeError:
            pdf_conf = conf.copy()

        # Aseguramos nombres coherentes
        pdf_conf.columns = ["real", "prediccion", "cantidad"]

        # 1) Agrupar por (real, prediccion) para evitar duplicados
        pdf_agg = (
            pdf_conf
            .groupby(["real", "prediccion"], as_index=False)["cantidad"]
            .sum()
        )

        # 2) Etiquetas legibles
        map_real = {
            0: "Real: 0 (no generó pedido)",
            1: "Real: 1 (sí generó pedido)",
        }
        map_pred = {
            0: "Predicho: 0 (no pedido)",
            1: "Predicho: 1 (pedido)",
        }

        pdf_agg["real_label"] = pdf_agg["real"].map(map_real)
        pdf_agg["pred_label"] = pdf_agg["prediccion"].map(map_pred)

        # 3) Pivot a matriz 2x2
        matriz = (
            pdf_agg
            .pivot(index="real_label", columns="pred_label", values="cantidad")
            .fillna(0)
        )

        fig_cm = px.imshow(
            matriz,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(
                x="Predicción del modelo",
                y="Etiqueta real",
                color="Cantidad"
            ),
            title="Matriz de confusión – modelo de predicción de pedidos"
        )
        fig_cm.update_layout(xaxis_side="top")

        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown(
            """
            <p style='font-size: 0.85rem; color: gray;'>
            La matriz de confusión permite evaluar qué tan bien el modelo distingue entre gestiones que generan pedido y las que no.<br>
            • El modelo clasifica correctamente la mayor parte de los casos sin pedido (≈34.9k aciertos).<br>
            • También identifica de forma adecuada los pedidos reales (≈22.9k aciertos).<br>
            • Los errores de predicción se mantienen moderados frente al volumen total analizado.<br>
            </p>
            """,
            unsafe_allow_html=True
        )

        # -------- PRUEBA EN VIVO DEL MODELO DE PEDIDOS --------
        st.markdown("---")
        st.subheader("✍️ Prueba en vivo del modelo de pedidos")

        st.caption("Configura una gestión hipotética y el modelo estimará la probabilidad de que genere un pedido.")

        if (
            "pedidos_pipeline" not in st.session_state
            or "pedidos_lr_model" not in st.session_state
        ):
            st.warning("Primero entrena el modelo con el botón de arriba.")
        else:
            pipeline_model = st.session_state["pedidos_pipeline"]
            lr_model = st.session_state["pedidos_lr_model"]

            with st.form("form_pred_pedido"):
                col1, col2 = st.columns(2)
                with col1:
                    medio = st.selectbox("Medio de contacto", ["llamada", "whatsapp"])
                    resultado_asesor = st.selectbox(
                        "Resultado del asesor",
                        ["interesado", "no_interesado"]
                    )
                with col2:
                    hora = st.slider("Hora del día", min_value=9, max_value=21, value=12)

                submitted = st.form_submit_button("Predecir probabilidad de pedido")

                if submitted:
                    # 🔹 Solo simulamos fecha ficticia para generar el datetime
                    fecha_hora_str = f"2025-01-01 {hora:02d}:00:00"

                    pred_clase, prob_pedido = predecir_pedido(
                        pipeline_model=pipeline_model,
                        modelo=lr_model,
                        spark=get_spark(),
                        medio=medio,
                        resultado_asesor=resultado_asesor,
                        fecha_hora_str=fecha_hora_str,
                    )

                    etiqueta = "Sí generaría pedido" if pred_clase == 1 else "No generaría pedido"

                    st.success(f"Predicción: **{etiqueta}**")
                    st.metric("Probabilidad estimada de pedido", f"{prob_pedido*100:.2f}%")

                    # 📌 Simulación por hora (sin usar fecha real)
                    horas = list(range(9, 22))
                    probs_horas = []

                    for h in horas:
                        fecha_hora_simulada = f"2025-01-01 {h:02d}:00:00"

                        _, prob_h = predecir_pedido(
                            pipeline_model=pipeline_model,
                            modelo=lr_model,
                            spark=get_spark(),
                            medio=medio,
                            resultado_asesor=resultado_asesor,
                            fecha_hora_str=fecha_hora_simulada
                        )
                        probs_horas.append(prob_h)

                    if not probs_horas:
                        st.warning("No se pudo calcular la probabilidad por hora.")
                    else:
                        probs_horas_np = np.array(probs_horas)
                        idx_orden = probs_horas_np.argsort()[::-1]
                        top_k = 3
                        mejores_idx = idx_orden[:top_k]
                        mejores_horas = [horas[i] for i in mejores_idx]
                        mejores_probs = [probs_horas_np[i] for i in mejores_idx]

                        st.markdown("### 🧭 Recomendación de horario:")

                        texto_mejores = ", ".join(
                            [f"{h:02d}:00 (~{p*100:.1f}%)" for h, p in zip(mejores_horas, mejores_probs)]
                        )

                        st.info(
                            f"Las horas con **mayor probabilidad de generar pedido** son: {texto_mejores}.<br>"
                            f"Seleccionaste **{hora:02d}:00**, con una probabilidad estimada de "
                            f"**{prob_pedido*100:.1f}%**."
                        )

                        # Mejoras sugeridas
                        st.subheader("💡 ¿Cómo podría mejorar esta probabilidad?")

                        otro_medio = "whatsapp" if medio == "llamada" else "llamada"
                        _, prob_otro_medio = predecir_pedido(
                            pipeline_model=pipeline_model,
                            modelo=lr_model,
                            spark=get_spark(),
                            medio=otro_medio,
                            resultado_asesor=resultado_asesor,
                            fecha_hora_str=fecha_hora_str,
                        )

                        recomendaciones = []

                        mejor_hora = mejores_horas[0]
                        mejor_prob = mejores_probs[0]

                        if mejor_hora != hora and mejor_prob > prob_pedido + 0.03:
                            recomendaciones.append(
                                f"- Reagendar alrededor de **{mejor_hora:02d}:00** "
                                f"(probabilidad estimada: **{mejor_prob*100:.1f}%**)."
                            )

                        if prob_otro_medio > prob_pedido + 0.03:
                            recomendaciones.append(
                                f"- Cambiar a **{otro_medio}** puede aumentar la probabilidad "
                                f"a **{prob_otro_medio*100:.1f}%**."
                            )

                        if not recomendaciones:
                            st.success(
                                "La configuración actual ya es bastante buena. "
                                "Puedes explorar el análisis por hora para optimizar más."
                            )
                        else:
                            st.markdown(
                                "Acciones que **podrían aumentar la probabilidad de generar pedido**:"
                            )
                            for rec in recomendaciones:
                                st.markdown(rec)

    else:
        st.warning("Pulsa el botón para entrenar el modelo de predicción de pedidos.")



# ===============================
# PÁGINA: DASHBOARD Y CONCLUSIONES
# ===============================
elif page == "📝 Conclusiones":
    st.title("📌 Conclusiones")

    st.info("""
    1. **Gestiones y eficiencia:** En 2025 se realizaron 1 millón de gestiones, pero solo el 40% concluyen en pedido, lo que evidencia un amplio margen de mejora comercial.

    2. **Patrón mensual:** Aunque las gestiones se mantienen estables, mayo destaca con el mayor número de pedidos aun sin ser el mes más activo. Importa más la calidad de la gestión que el volumen.

    3. **Interés del cliente:** La mayoría se clasifica como no interesado, pero el comportamiento es estable. Esto refuerza que el interés depende del contenido de la gestión y no del número.

    4. **Medio vs categoría:** Las llamadas son consistentemente más efectivas que WhatsApp. La categoría con mayor conversión es conectividad.  
    → **Mejor combinación: llamada + conectividad.**

    5. **Rendimiento de asesores:** Los mejores asesores generan ~1000 pedidos, mientras que los de menor desempeño rondan los ~550, pese a tener volúmenes similares. Hay oportunidad de replicar buenas prácticas.

    6. **Modelo de predicción:** La regresión logística logra un 72% de accuracy, permitiendo estimar si una gestión terminará en pedido y en qué condiciones.

    7. **Modelo NLP:** El modelo NLP obtiene 92% de accuracy y detecta patrones lingüísticos que predicen el verdadero interés del cliente, útil para entrenamiento y estandarización de notas.
    """)