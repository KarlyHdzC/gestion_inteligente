# -------------------------------------------------------------------------------------
# OBJETIVO: CARGAR LA INFORMACIÓN DEL ARCHIVO gestiones.csv 
# CON LA INFORMACIÓN DE GESTIONES REALIZADAS QUE GENERAN PEDIDOS 
# -------------------------------------------------------------------------------------
# Es importante que los campos estén delineados con el tipo de dato específico
# pedido_generado → int (0/1)
# monto_pedido → double
# fecha_hora → timestamp
# id_pedido → string
# nota → string

# Importar librerías necesarias
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when,
    lit,
    coalesce,
    to_timestamp
)
import sys

# Detecta la ruta raíz del proyecto automáticamente
ruta = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder = os.path.join(ruta, "datos")

# Crear una sesión de Spark
def crear_spark():
    python_exec = sys.executable

    # Forzar que tanto el driver como los workers usen el mismo intérprete de Python
    os.environ["PYSPARK_PYTHON"] = python_exec
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_exec

    spark = (
        SparkSession.builder
        .appName("gestion_inteligente")
        # Forzar versión de Python para los workers de Spark
        .config("spark.executorEnv.PYSPARK_PYTHON", python_exec)
        .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON", python_exec)
        .getOrCreate()
    )
    return spark

def obtener_contexto(spark): 
    sc = spark.sparkContext
    print(f"✅ SparkSession creada: {spark}")
    print(f"✅ SparkContext disponible: {sc}")
    print(f"📊 Versión de Spark: {spark.version}")
    print(f"🔗 Spark UI disponible en: {spark.sparkContext.uiWebUrl}")

def cargar_gestiones(spark, ruta_csv: str):
    """
    Lee el CSV de gestiones, castea tipos y aplica limpieza + reglas de negocio
    para que el DataFrame ya regrese 'listo para usar' en Streamlit.

    Reglas aplicadas:
    - Casteo de:
        * pedido_generado -> int
        * monto_pedido    -> double
        * fecha_hora      -> timestamp

    - Relleno de nulos en:
        * id_cliente        -> 'C000000000'
        * id_asesor         -> 'A000000000'
        * medio             -> 'desconocido'
        * resultado_asesor  -> 'no_especificado'
        * nota              -> 'Sin nota capturada'
        * fecha_hora        -> '2025-01-01 00:00:00' (si viene nula)

    - Reglas de negocio según pedido_generado:
        * Si pedido_generado = 0:
            - id_pedido          = 'NA'
            - monto_pedido       = 0
            - producto_categoria = 'NA'
        * Si pedido_generado = 1:
            - id_pedido nulo          -> 'P000000000'
            - monto_pedido nulo       -> 0
            - producto_categoria nulo -> 'NA'
    """

    # 1) Lectura base
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(ruta_csv)
    )

    # 2) Casteo de tipos principales
    df = (
        df
        .withColumn("pedido_generado", col("pedido_generado").cast("int"))
        .withColumn("monto_pedido", col("monto_pedido").cast("double"))
        .withColumn("fecha_hora", col("fecha_hora").cast("timestamp"))
    )

    # 3) Relleno general de nulos en columnas "simples"
    df = df.fillna({
        "id_cliente": "C000000000",
        "id_asesor": "A000000000",
        "medio": "desconocido",
        "resultado_asesor": "no_especificado",
        "nota": "Sin nota capturada"
    })

    # 4) Relleno de nulos en fecha_hora
    df = df.withColumn(
        "fecha_hora",
        coalesce(
            col("fecha_hora"),
            to_timestamp(lit("2025-01-01 00:00:00"), "yyyy-MM-dd HH:mm:ss")
        )
    )

    # 5) Aplicar reglas de negocio según pedido_generado
    # id_pedido
    df = df.withColumn(
        "id_pedido",
        when(col("pedido_generado") == 0, lit("NA"))
        .otherwise(coalesce(col("id_pedido"), lit("P000000000")))
    )

    # monto_pedido
    df = df.withColumn(
        "monto_pedido",
        when(col("pedido_generado") == 0, lit(0.0))
        .otherwise(coalesce(col("monto_pedido"), lit(0.0)))
    )

    # producto_categoria
    df = df.withColumn(
        "producto_categoria",
        when(col("pedido_generado") == 0, lit("NA"))
        .otherwise(coalesce(col("producto_categoria"), lit("NA")))
    )

    return df
