from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    StringIndexer,
    NGram,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import re
import unicodedata

# ==========================================================
#  🔧 Función de limpieza reutilizable (Python + UDF Spark)
# ==========================================================
def limpiar_texto(texto: str) -> str:
    if texto is None:
        return ""

    # 1) Minúsculas
    texto = texto.lower()

    # 2) Quitar acentos
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")

    # 3) Quitar caracteres raros / signos de puntuación
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)

    # 4) Quitar números
    texto = re.sub(r"\d+", " ", texto)

    # 5) Espacios repetidos
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def limpiar_texto_basico_py(texto: str) -> str:
    """
    Limpieza básica de texto:
    - Minúsculas
    - Mantiene ñ y acentos
    - Quita símbolos raros
    - Colapsa espacios múltiples
    """
    if texto is None:
        return ""

    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


limpiar_texto_udf = udf(limpiar_texto_basico_py, StringType())

# ==========================================================
#  🧠 Entrenamiento del modelo NLP
# ==========================================================
def entrenar_modelo_nlp(df, ruta_modelo: str | None = None):
    """
    Entrena un modelo NLP para clasificar notas de asesores según 'resultado_asesor'.
    Entrada: df con columnas "nota" y "resultado_asesor".
    Usa 'nota_clean' si ya existe; si no, la genera.
    """

    # Si el DF no tiene 'nota_clean', la generamos
    if "nota_clean" not in df.columns:
        df = df.withColumn("nota_clean", limpiar_texto_udf(col("nota")))

    data = df.select("nota_clean", "resultado_asesor").na.drop()

    # 1) Indexado de etiqueta
    label_indexer = StringIndexer(
        inputCol="resultado_asesor",
        outputCol="label"
    )

    # 2) Tokenización robusta
    tokenizer = RegexTokenizer(
        inputCol="nota_clean",
        outputCol="tokens",
        pattern=r"\W+",
        minTokenLength=2
    )

    # 3) Stopwords español + personalizadas
    # Conserva negaciones comunes para que el modelo aprenda tono negativo
    stopwords_es = StopWordsRemover.loadDefaultStopWords("spanish")
    stopwords_es = [w for w in stopwords_es if w not in ["no", "nunca", "ni"]]

    stopwords_extra = [
    # Verbos repetidos en ambos casos
    "dijo", "comento", "menciono", "indico",
    "pidio", "mostro", "solicito", "confirmo",
    "expreso", "expresó",

    # Palabras estructurales
    "que", "de", "del", "la", "el", "al", "por", "para",
    "en", "con", "se", "su", "ya", "solo", "le",

    # Contexto genérico del proceso
    "cliente", "asesor", "gestion", "llamada", "contacto",
    "informacion", "detalle", "detalles", "opcion", "opciones",

    # Productos y categorías generales (se repiten mucho)
    "paquete", "variante", "variantes", "producto", "productos",
    "hogar", "movilidad", "conectividad",

    # Conversación general que no discrimina
    "propuesta", "oferta", "compra", "comprar", "adquirir",
    "reservar", "precio", "precios", "servicio", "servicios",
    "utiliza", "necesita", "necesario",

    # Palabras repetitivas y neutrales
    "cuando", "como", "mas", "más", "tambien", "también",
    "proximo", "proxima", "próximo", "próxima",
    "nuevamente", "probablemente", "estuvo", "estaba",
    "cuenta", "aclaro", "aclaró", "similar", "tiene",
    "decia", "decía", "podria", "podría"
    ]

    stopwords_es.extend(stopwords_extra)
    stopwords_es = list(set([w.strip() for w in stopwords_es]))

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="tokens_clean",
        stopWords=stopwords_es
    )

    # 4) N-gramas (opcional)
    ngrammer = NGram(
        n=1,
        inputCol="tokens_clean",
        outputCol="tokens_lemma"
    )

    # 5) Vectorización
    vectorizer = CountVectorizer(
        inputCol="tokens_lemma",
        outputCol="tf"
    )

    # 6) TF-IDF
    idf = IDF(
        inputCol="tf",
        outputCol="features"
    )

    # 7) Clasificador
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20
    )

    pipeline = Pipeline(stages=[
        label_indexer,
        tokenizer,
        remover,
        ngrammer,
        vectorizer,
        idf,
        lr
    ])

    # Split train-test
    # Fijamos semilla para mantener resultados reproducibles en dashboards
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    num_train = train.count()
    num_test = test.count()

    # Entrenar modelo
    modelo = pipeline.fit(train)

    # Evaluar
    pred = modelo.transform(test)

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    accuracy = evaluator_acc.evaluate(pred)
    f1 = evaluator_f1.evaluate(pred)

    # Matriz de confusión
    confusion_df = (
        pred.groupBy("label", "prediction")
            .count()
            .orderBy("label", "prediction")
    )

    labels = modelo.stages[0].labels

    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "num_train": num_train,
        "num_test": num_test,
        "labels": labels
    }

    # Guardar modelo si se especificó ruta
    if ruta_modelo:
        modelo.write().overwrite().save(ruta_modelo)

    return metrics, confusion_df, modelo
