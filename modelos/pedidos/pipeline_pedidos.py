# modelos/pedidos/pipeline_pedidos.py

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)


def crear_pipeline_pedidos():
    """
    Pipeline de features para predecir `pedido_generado` usando SOLO variables
    estructuradas (nada de texto):

    - medio (categórica)
    - resultado_asesor (categórica)
    - hora (numérica)
    - dia_semana (numérica)
    - mes (numérica)

    La columna `nota` sigue existiendo en el DataFrame, pero NO se usa en el
    pipeline de este modelo.
    """

    # Categóricas
    idx_medio = StringIndexer(
        inputCol="medio",
        outputCol="medio_idx",
        handleInvalid="keep",
    )
    idx_resultado = StringIndexer(
        inputCol="resultado_asesor",
        outputCol="resultado_idx",
        handleInvalid="keep",
    )

    oh_medio = OneHotEncoder(
        inputCol="medio_idx",
        outputCol="medio_vec",
    )
    oh_resultado = OneHotEncoder(
        inputCol="resultado_idx",
        outputCol="resultado_vec",
    )

    # Ensamblar todas las features estructuradas en un solo vector numérico
    assembler = VectorAssembler(
        inputCols=[
            "medio_vec",
            "resultado_vec",
            "hora",
            "dia_semana",
            "mes",
        ],
        outputCol="features",
    )

    pipeline = Pipeline(
        stages=[
            idx_medio,
            idx_resultado,
            oh_medio,
            oh_resultado,
            assembler,
        ]
    )

    return pipeline
