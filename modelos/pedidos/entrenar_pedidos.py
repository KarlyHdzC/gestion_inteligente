from pyspark.sql.functions import hour, dayofweek, month, col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from .pipeline_pedidos import crear_pipeline_pedidos


def _metricas_desde_pred(pred_df, label_col="pedido_generado"):
    """Calcula accuracy, f1, precision, recall, auc."""
    eval_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy",
    )
    eval_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1",
    )
    eval_prec = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="precisionByLabel",
    )
    eval_rec = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="recallByLabel",
    )
    eval_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    return {
        "accuracy": float(eval_acc.evaluate(pred_df)),
        "f1":       float(eval_f1.evaluate(pred_df)),
        "precision":float(eval_prec.evaluate(pred_df)),
        "recall":   float(eval_rec.evaluate(pred_df)),
        "auc":      float(eval_auc.evaluate(pred_df)),
    }


def entrenar_modelos_pedidos(df):
    """
    Entrena un modelo de Regresión Logística para predecir `pedido_generado`
    usando SOLO variables estructuradas:

    - medio
    - resultado_asesor
    - hora
    - dia_semana
    - mes

    La columna `nota` NO se usa aquí (queda solo para el modelo NLP).
    """

    # Asegurar tipo entero para la etiqueta
    df = df.withColumn("pedido_generado", col("pedido_generado").cast("int"))

    # Agregar columnas temporales desde fecha_hora
    df_feat = (
        df.withColumn("hora", hour("fecha_hora"))
          .withColumn("dia_semana", dayofweek("fecha_hora"))
          .withColumn("mes", month("fecha_hora"))
    )

    # (Opcional) Submuestreo para evitar problemas de memoria en local;
    # ajusta la fracción según la capacidad de tu equipo.
    df_feat = df_feat.sample(withReplacement=False, fraction=0.4, seed=42)

    # Split train-test
    train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
    num_train = train_df.count()
    num_test = test_df.count()

    # Pipeline de features estructuradas
    pipeline = crear_pipeline_pedidos()
    pipeline_model = pipeline.fit(train_df)

    train_feat = pipeline_model.transform(train_df)
    test_feat = pipeline_model.transform(test_df)

    # Modelo: Regresión Logística
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="pedido_generado",
        maxIter=30,
    )

    lr_model = lr.fit(train_feat)  # Entrenamos solo con features estructuradas

    # Predicciones sobre test
    pred_lr = lr_model.transform(test_feat)

    # Métricas
    m_lr = _metricas_desde_pred(pred_lr)

    # Matriz de confusión
    cm_lr = (
        pred_lr.groupBy("pedido_generado", "prediction")
               .count()
               .orderBy("pedido_generado", "prediction")
               .toPandas()
               .rename(columns={
                   "pedido_generado": "real",
                   "prediction": "prediccion",
                   "count": "cantidad",
               })
    )

    m_lr["num_train"] = num_train
    m_lr["num_test"] = num_test
    m_lr["confusion"] = cm_lr

    resultados = {
        "lr": m_lr,
    }

    return resultados, pipeline_model, lr_model
