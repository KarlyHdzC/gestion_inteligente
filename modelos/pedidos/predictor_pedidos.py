from pyspark.sql.functions import (
    to_timestamp,
    hour,
    dayofweek,
    month,
    col,
)


def predecir_pedido(
    pipeline_model,
    modelo,
    spark,
    medio: str,
    resultado_asesor: str,
    fecha_hora_str: str,
):
    """
    Predice si una gestión generaría un pedido (0/1) y su probabilidad,
    usando SOLO variables estructuradas.

    Parámetros
    ----------
    pipeline_model : PipelineModel
        Pipeline ya entrenado (crear_pipeline_pedidos + fit).
    modelo : LogisticRegressionModel (u otro clasificador)
        Modelo entrenado sobre la columna 'features'.
    spark : SparkSession
        Sesión de Spark activa.
    medio : str
        Medio de contacto, por ejemplo: 'llamada' o 'whatsapp'.
    resultado_asesor : str
        Resultado del asesor: 'interesado' o 'no_interesado'.
    fecha_hora_str : str
        Fecha y hora de la gestión en formato 'YYYY-MM-DD HH:MM:SS'.

    Returns
    -------
    pred_clase : int
        Predicción de clase: 0 = no genera pedido, 1 = genera pedido.
    prob_pedido : float
        Probabilidad estimada de generar pedido (clase 1).
    """

    # 1) Construimos un DataFrame de una sola fila con los datos de entrada
    data = [(medio, resultado_asesor, fecha_hora_str)]
    df_input = spark.createDataFrame(
        data,
        ["medio", "resultado_asesor", "fecha_hora_str"]
    )

    # 2) Convertimos el string de fecha a timestamp real
    df_input = df_input.withColumn(
        "fecha_hora",
        to_timestamp(col("fecha_hora_str"), "yyyy-MM-dd HH:mm:ss")
    )

    # 3) Derivamos las variables temporales igual que en el entrenamiento
    df_feat = (
        df_input
        .withColumn("hora", hour(col("fecha_hora")))
        .withColumn("dia_semana", dayofweek(col("fecha_hora")))
        .withColumn("mes", month(col("fecha_hora")))
    )

    # 4) Aplicamos el pipeline de features (indexadores, one-hot, assembler)
    df_feat_transf = pipeline_model.transform(df_feat)

    # 5) Aplicamos el modelo de clasificación (Regresión Logística)
    df_pred = modelo.transform(df_feat_transf).select("prediction", "probability")

    # Solo una fila; se recoge para extraer clase y vector de probabilidades
    fila = df_pred.collect()[0]
    pred_clase = int(fila["prediction"])
    prob_vector = fila["probability"]

    # Para binario, el índice 1 es la probabilidad de clase "1" (pedido)
    prob_pedido = float(prob_vector[1])

    return pred_clase, prob_pedido
