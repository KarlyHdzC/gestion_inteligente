# ⭐️ **Gestión Inteligente**  
### *Análisis Comercial · NLP con PySpark · Modelos Predictivos · Streamlit Dashboard*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PySpark-MLlib-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

---

# 📘 **Descripción General**

**Gestión Inteligente** es un sistema completo para:

✔️ Analizar gestiones comerciales  
✔️ Detectar intención real del cliente mediante **NLP**  
✔️ Predecir la probabilidad de generar un pedido  
✔️ Recomendar el mejor horario para realizar gestiones  
✔️ Visualizar todo en un **dashboard con Streamlit**

Incluye:

- Procesamiento masivo con **PySpark**
- Limpieza avanzada del texto (acentos, minúsculas, símbolos)
- Modelos ML (Regresión Logística & Random Forest)
- Wordclouds y análisis exploratorios interactivos
- Recomendación inteligente de horarios

---

# 🧠 **Arquitectura del Proyecto**

```
gestion_inteligente/
│
├── datos/
│     └── gestiones.csv
│
├── modelos/
│     ├── modelo_nlp.py
│     └── pedidos/
│           ├── pipeline_pedidos.py
│           ├── entrenar_pedidos.py
│           └── predictor_pedidos.py
│
├── src/
│     └── lectura_gestiones.py
│
├── app_streamlit.py
└── README.md
```

---

# 🚀 **Características Principales**

## 📊 **1) Análisis Exploratorio (EDA)**
Incluye visualizaciones profesionales:

- Gestiones por hora, día y mes  
- Conversión por medio (llamada / WhatsApp)  
- Categorías con más ventas  
- Top & Bottom 10 asesores  
- Heatmaps  
- Wordclouds 
- Comparaciones entre interés y comportamiento temporal  

---

## 📝 **2) Modelo NLP – Clasificación de notas**
Determina si una nota corresponde a un cliente **interesado** o **no_interesado**.

### 👉 Pipeline implementado:
- Limpieza avanzada: minúsculas, acentos, signos, normalización  
- Tokenización  
- Stopwords extendidas  
- TF-IDF  
- Regresión Logística  
- Matriz de confusión  
- Prueba en vivo  

📈 *Accuracy típico: ~92%*

---

## 🔮 **3) Modelo ML – Predicción de Pedidos**
Usa variables como:

- medio  
- resultado_asesor  
- hora  

Modelo incluidos:

| Modelo | Métrica | Uso |
|--------|---------|------|
| **Regresión Logística** | Accuracy ~72% | Modelo principal |

---

## ⏰ **4) Recomendación de Mejor Horario**
El sistema simula la gestión en cada hora entre **09:00 y 21:00** y devuelve:

- Top 3 horarios con mayor conversión  
- Comparación con la hora elegida  
- Recomendación de cambio de medio  

---

# 🖥️ **Cómo Ejecutar el Proyecto**

### 1️⃣ Activar entorno
```bash
conda activate streamlit_pyspark
```

### 2️⃣ Ejecutar interfaz
```bash
streamlit run app_streamlit.py
```

## Si no se cuenta con el entorno: 
## Cómo reproducir el proyecto

1. Clonar el repositorio.
2. Crear y activar un entorno virtual.
3. Instalar dependencias:
   pip install -r requirements.txt
4. Ejecutar la app:
   streamlit run app_streamlit.py

---

# 📁 **Estructura Técnica**

| Archivo | Descripción |
|--------|-------------|
| `modelo_nlp.py` | Pipeline completo de NLP |
| `pipeline_pedidos.py` | Feature engineering del modelo ML |
| `entrenar_pedidos.py` | Entrenamiento y evaluación de modelos |
| `predictor_pedidos.py` | Predicciones en vivo |
| `app_streamlit.py` | Dashboard principal |
| `lectura_gestiones.py` | Validación de datos |

---

# 🌟 **Resultados Destacados**

### ✔️ Análisis comercial
- 1 millón de gestiones analizadas  
- 40% terminan en pedido  
- Llamada > WhatsApp en efectividad  
- Conectividad es la categoría más rentable  
- Alto contraste entre top/bottom asesores  

### ✔️ Modelo NLP
- Accuracy de 92%  
- Identificación clara del lenguaje de compra  
- Limpieza robusta del texto  

### ✔️ Modelo de predicción
- Accuracy: ~72%  
- Recomendaciones accionables por horario  
- Comparación entre medios  

---

# 📌 **Notas Técnicas Importantes**

- TF-IDF no interpreta semántica profunda  
- Streamlit cachea wordclouds para evitar recálculos  
- PySpark requiere versiones compatibles de Python y Java  
- El dataset usado es **sintético pero con reglas reales de comportamiento comercial**  

---

# 🏁 **Conclusión**

**Gestión Inteligente** combina:

- Ciencia de datos  
- Machine Learning  
- NLP  
- Big Data  
- Visualización interactiva  

Para transformar gestiones comerciales en decisiones accionables.

El resultado es una herramienta moderna, escalable y lista para integrarse en una operación real de ventas o atención al cliente.