import numpy as np
import pandas as pd

# ------------------------------------------------------
#                CONFIGURACIÓN GENERAL
# ------------------------------------------------------

n = 1_000_000            # <<< Tamaño del dataset sintético; ajusta según tu equipo
np.random.seed(42)       # Reproducibilidad

def format_id(prefix, numbers):
    """Genera IDs tipo C000000001, A000000234, P000001231."""
    return [f"{prefix}{int(x):09d}" for x in numbers]


# ------------------------------------------------------
#              1) CLIENTES Y ASESORES
# ------------------------------------------------------

client_pool = np.arange(1, 200_001)   # 200 mil clientes posibles
advisor_pool = np.arange(1, 501)      # 500 asesores

id_cliente_nums = np.random.choice(client_pool, size=n, replace=True)
id_asesor_nums  = np.random.choice(advisor_pool, size=n, replace=True)

id_clientes = format_id("C", id_cliente_nums)
id_asesores = format_id("A", id_asesor_nums)

# Factores latentes de "propensión a comprar" por cliente y habilidad de asesor
cliente_factor = np.random.normal(0, 0.05, size=client_pool.size)
asesor_factor  = np.random.normal(0, 0.05, size=advisor_pool.size)

cliente_factor_map = dict(zip(client_pool, cliente_factor))
asesor_factor_map  = dict(zip(advisor_pool, asesor_factor))

cliente_f = np.vectorize(cliente_factor_map.get)(id_cliente_nums)
asesor_f  = np.vectorize(asesor_factor_map.get)(id_asesor_nums)


# ------------------------------------------------------
#           2) FECHA Y HORA 2025 (9:00–21:00)
# ------------------------------------------------------

# Fechas aleatorias en 2025 (solo día)
start_date = pd.Timestamp("2025-01-01")
end_date   = pd.Timestamp("2025-11-22")   # puedes extender a 2025-12-31 si quieres todo el año
days_range = (end_date - start_date).days + 1

rand_days = np.random.randint(0, days_range, size=n)
dates = start_date + pd.to_timedelta(rand_days, unit="D")

# Hora de la gestión SOLO entre 09:00 y 21:00
horas = np.random.randint(9, 22, size=n)       # 9..21 inclusive
minutos = np.random.randint(0, 60, size=n)
segundos = np.random.randint(0, 60, size=n)

fecha_hora = (
    dates
    + pd.to_timedelta(horas, unit="h")
    + pd.to_timedelta(minutos, unit="m")
    + pd.to_timedelta(segundos, unit="s")
)

# Índice de tiempo
dt = pd.to_datetime(fecha_hora)  # DatetimeIndex

# Componentes de tiempo (OJO: aquí ya NO usamos .dt)
horas       = dt.hour.values
meses       = dt.month.values
dias_semana = dt.dayofweek.values  # 0 = lunes, 6 = domingo
dias_mes    = dt.day.values


# ------------------------------------------------------
#                3) MEDIO DE CONTACTO
# ------------------------------------------------------

medios = np.random.choice(
    ["llamada", "whatsapp"],
    size=n,
    p=[0.55, 0.45]
)


# ------------------------------------------------------
#            4) RESULTADO DEL ASESOR
# ------------------------------------------------------

resultado_asesor = np.random.choice(
    ["interesado", "no_interesado"],
    size=n,
    p=[0.45, 0.55]  # proporción de interesados vs no interesados
)


# ------------------------------------------------------
#   5) PATRÓN REALISTA DE PROBABILIDAD DE PEDIDO (p)
# ------------------------------------------------------
# Inspirado en patrones de e-commerce:
# - interesados convierten mucho más que no_interesados
# - WhatsApp convierte mejor que llamada
# - Picos en 10–13h y 17–21h, baja en 9 y 14–16
# - Lunes/Martes flojos, Jueves/Viernes fuertes, domingo sube un poco
# - Ene/Feb bajo, Mayo (Hot Sale), Nov (Buen Fin) y Dic más altos
# - Efecto quincena (1–3, 14–17, 29–31)
# - Efecto cliente y asesor (latentes)

# Base por interés
prob = np.where(resultado_asesor == "interesado", 0.55, 0.12).astype(float)

# Ajuste por medio (WhatsApp suele convertir mejor)
prob += np.where(medios == "whatsapp", 0.05, 0.0)

# Ajuste por hora del día
#  9h: un poco más bajo (apenas apertura)
prob += np.where(horas == 9, -0.02, 0.0)
# 10–13h: primer pico fuerte
prob += np.where((horas >= 10) & (horas <= 13), 0.06, 0.0)
# 14–16h: descanso / comida, baja un poco
prob += np.where((horas >= 14) & (horas <= 16), -0.02, 0.0)
# 17–21h: segundo pico fuerte
prob += np.where((horas >= 17) & (horas <= 21), 0.07, 0.0)

# Ajuste por día de la semana (0=L, 6=D)
# Lunes/Martes un poco más flojos
prob += np.where(np.isin(dias_semana, [0, 1]), -0.02, 0.0)
# Jueves/Viernes más fuertes
prob += np.where(np.isin(dias_semana, [3, 4]), 0.03, 0.0)
# Domingo: ligero boost general
prob += np.where(dias_semana == 6, 0.02, 0.0)

# Ajuste por mes (estacionalidad)
# Ene-Feb: más flojos
prob += np.where(np.isin(meses, [1, 2]), -0.03, 0.0)
# Mayo (Hot Sale) y Nov (Buen Fin)
prob += np.where(meses == 5, 0.05, 0.0)   # Hot Sale
prob += np.where(meses == 11, 0.07, 0.0)  # Buen Fin
# Dic: Navidad / fin de año
prob += np.where(meses == 12, 0.05, 0.0)

# Efecto quincena (1–3, 14–17, 29–31)
mask_quincena = (
    (dias_mes <= 3)
    | ((dias_mes >= 14) & (dias_mes <= 17))
    | (dias_mes >= 29)
)
prob += np.where(mask_quincena, 0.04, 0.0)

# Efecto cliente y asesor (latentes)
prob += cliente_f
prob += asesor_f

# Clip para mantener probabilidades en [0.01, 0.95]
prob = np.clip(prob, 0.01, 0.95)

# Muestra final de pedidos generados (0/1)
pedido_generado = np.random.binomial(1, prob, size=n)


# ------------------------------------------------------
#  6) CATEGORÍA SOLO SI HAY PEDIDO, CON PATRÓN POR HORA
# ------------------------------------------------------
# Patrón realista aproximado:
# - Conectividad: más en horario laboral y días hábiles
# - Movilidad: más tarde y fines de semana
# - Hogar: noches y fines de semana, muy fuerte en domingo y en Nov/Dic

producto_categoria = np.full(n, "NA", dtype=object)

mask_pedido = pedido_generado == 1
num_pedidos = int(mask_pedido.sum())

if num_pedidos > 0:
    horas_p = horas[mask_pedido]
    dias_p = dias_semana[mask_pedido]
    meses_p = meses[mask_pedido]

    # Pesos base
    w_con = np.full(num_pedidos, 0.40)  # conectividad
    w_mov = np.full(num_pedidos, 0.30)  # movilidad
    w_hog = np.full(num_pedidos, 0.30)  # hogar

    # Horario laboral (9–17) en días hábiles (0–4): boost conectividad
    mask_work = (horas_p >= 9) & (horas_p <= 17) & (dias_p < 5)
    w_con += 0.15 * mask_work

    # Noches (18–21): movilidad y hogar suben
    mask_night = (horas_p >= 18) & (horas_p <= 21)
    w_mov += 0.10 * mask_night
    w_hog += 0.05 * mask_night

    # Fines de semana (5=Sábado, 6=Domingo): movilidad y hogar ganan peso
    mask_weekend = dias_p >= 5
    w_mov += 0.05 * mask_weekend
    w_hog += 0.05 * mask_weekend

    # Domingo: hogar aún más fuerte
    mask_domingo = dias_p == 6
    w_hog += 0.08 * mask_domingo

    # Nov/Dic: hogar + conectividad fuertes
    mask_nov_dic = np.isin(meses_p, [11, 12])
    w_hog += 0.05 * mask_nov_dic
    w_con += 0.03 * mask_nov_dic

    # Normalizar a probabilidades
    sum_w = w_con + w_mov + w_hog
    p_con = w_con / sum_w
    p_mov = w_mov / sum_w
    p_hog = w_hog / sum_w

    # Muestreo por fila usando umbrales acumulados
    u = np.random.rand(num_pedidos)
    cat_pedidos = np.empty(num_pedidos, dtype=object)

    thr_con = p_con
    thr_mov = p_con + p_mov  # corte para movilidad

    cat_pedidos[u < thr_con] = "conectividad"
    cat_pedidos[(u >= thr_con) & (u < thr_mov)] = "movilidad"
    cat_pedidos[u >= thr_mov] = "hogar"

    producto_categoria[mask_pedido] = cat_pedidos


# ------------------------------------------------------
#           7) PLANTILLAS COMPLETAS DE NOTAS
# ------------------------------------------------------

templates_interes = [
    "Cliente muy interesado en productos de {cat}, pidió más detalles sobre promociones.",
    "Mostró intención de compra de {cat}, comentó que decidirá en la próxima quincena.",
    "Cliente receptivo, comparó precios y dijo que probablemente compre {cat}.",
    "Solicitó información adicional de {cat} y preguntó por métodos de pago.",
    "Cliente satisfecho con la explicación de {cat}, pidió mantener contacto.",
    "Pidió cotización formal de {cat} y mencionó que necesita el producto pronto.",
    "Cliente dijo que le gustó la oferta de {cat} y que lo hablará con su familia.",
    "Mostró interés moderado por {cat}, pidió que le envíen el catálogo por {medio}.",
    "Cliente expresó fuerte interés en {cat}, pidió más detalles por {medio}.",
    "Comentó que estuvo buscando un producto de {cat} y que esta oferta le llamó la atención.",
    "Cliente mencionó que necesita {cat} pronto y que la propuesta le pareció muy competitiva.",
    "Pidió que lo contacten nuevamente por {medio} para confirmar la compra.",
    "Mostró entusiasmo cuando se le mencionó {cat}.",
    "Solicitó variantes del modelo de {cat} para comparar.",
    "Cliente dijo que {cat} es justo lo que estaba buscando.",
    "Pidió reservar la oferta de {cat}.",
    "Cliente pidió demostración de {cat}.",
    "Mostró disposición inmediata para adquirir {cat}.",
    "Cliente desea cerrar la compra de {cat} tan pronto como sea posible."
]

templates_no_interes = [
    "Cliente indicó que no le interesa {cat} y no desea más llamadas.",
    "Comentó que ya cuenta con un servicio similar y no requiere {cat}.",
    "Cliente mencionó que no tiene presupuesto para {cat}.",
    "Pidió cancelar futuras gestiones sobre {cat}.",
    "Cliente dijo que no ve necesario contratar {cat}.",
    "No mostró interés en {cat}, solo escuchó la propuesta.",
    "Cliente molesto por contacto recurrente.",
    "Dijo que no utiliza servicios de {cat}.",
    "No tiene pensado adquirir {cat} en el corto plazo.",
    "No considera prioritaria la compra de {cat}.",
    "Cliente no desea comprometerse con servicios de {cat}.",
    "No encuentra utilidad en adquirir {cat}.",
    "Recientemente compró algo similar a {cat}.",
    "No desea evaluar opciones de {cat}.",
    "Prefiere no recibir más información sobre {cat}."
]

templates_neutral = [
    "Cliente pidió volver a contactarlo en otro horario.",
    "Llamada breve, no se determinó interés.",
    "Cliente ocupado, pidió reagendar.",
    "Comunicación interrumpida, gestión inconclusa.",
    "Cliente pidió información general sin profundizar.",
    "No se percibió interés ni rechazo.",
    "Cliente solicitó que la información se enviara por {medio}.",
    "No fue posible evaluar si estaba interesado.",
    "Cliente tenía prisa, conversación corta.",
    "La comunicación fue cordial pero sin señales claras.",
    "Cliente pidió revisar la propuesta más tarde.",
    "Dificultad técnica, no se completó la llamada.",
    "Cliente escuchó pero no comentó sobre {cat}.",
    "Pidió que se le contacte otro día.",
    "Cliente no expresó ninguna intención clara."
]

templates_inconsistentes_interes = [
    "Aunque aparece como interesado, dijo que no planea comprar {cat}.",
    "Cliente mencionó que probablemente no comprará {cat}.",
    "Marcado como interesado, pero comentó que no lo necesita.",
    "Cliente dijo que no tiene intención real de comprar {cat}.",
    "Dijo que quizá compre {cat}, pero lo ve poco probable.",
    "Aunque figura como interesado, pidió no recibir más información.",
    "Mostró inseguridad sobre {cat}.",
    "Dijo que no confía en productos de {cat} pese a la clasificación.",
    "Comentó que no es buen momento para adquirir {cat}.",
    "Dijo que no tiene presupuesto para {cat}.",
    "Aunque aparece como interesado, no mostró interés real.",
    "Cliente expresó dudas fuertes sobre {cat}.",
    "Mencionó que tal vez más adelante, pero no ahora.",
    "Pidió no continuar el proceso aunque fue clasificado como interesado.",
    "El cliente no mostró señales reales de compra."
]

templates_inconsistentes_no_interes = [
    "Marcado como no interesado, pero pidió precios de {cat}.",
    "Dijo que quizá compre {cat} en la próxima quincena.",
    "Pidió catálogo y costos de {cat}.",
    "Mostró interés en promociones de {cat}.",
    "Solicitó que lo contacten más tarde para ver opciones.",
    "Preguntó por financiamiento para {cat}.",
    "Pidió revisar modelos de {cat}.",
    "Mostró curiosidad por beneficios de {cat}.",
    "Dijo que podría estar interesado dependiendo del precio.",
    "Solicitó comparativo de modelos de {cat}.",
    "Preguntó disponibilidad de {cat}.",
    "Pidió información detallada por {medio}.",
    "Dijo que evalúa comprar {cat} pronto.",
    "Mostró interés moderado aunque marcado como no interesado.",
    "Pidió saber cuándo habrá nuevas promociones."
]


# ------------------------------------------------------
#           8) GENERAR NOTAS (SIN 'NA' EN TEXTO)
# ------------------------------------------------------

notas = []
for i in range(n):
    medio_i = medios[i]
    res = resultado_asesor[i]
    tiene_pedido = pedido_generado[i] == 1
    r = np.random.rand()

    if res == "interesado":
        if r < 0.75:
            template = np.random.choice(templates_interes)
        elif r < 0.90:
            template = np.random.choice(templates_neutral)
        else:
            template = np.random.choice(templates_inconsistentes_interes)
    else:
        if r < 0.75:
            template = np.random.choice(templates_no_interes)
        elif r < 0.90:
            template = np.random.choice(templates_neutral)
        else:
            template = np.random.choice(templates_inconsistentes_no_interes)

    if tiene_pedido:
        cat = producto_categoria[i]
        nota = template.format(cat=cat, medio=medio_i)
    else:
        template_sin_cat = (
            template
            .replace(" de {cat}", "")
            .replace("{cat}", "")
        )
        template_sin_cat = " ".join(template_sin_cat.split())
        nota = template_sin_cat.format(medio=medio_i)

    nota = nota.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
    nota = " ".join(nota.split())
    notas.append(nota)


# ------------------------------------------------------
#        9) CREAR DATAFRAME PRINCIPAL
# ------------------------------------------------------

df = pd.DataFrame({
    "id_gestion": np.arange(1, n + 1),
    "id_cliente": id_clientes,
    "id_asesor": id_asesores,
    "medio": medios,
    "resultado_asesor": resultado_asesor,
    "fecha_hora": fecha_hora.astype("datetime64[s]"),
    "nota": notas,
    "pedido_generado": pedido_generado,
    "producto_categoria": producto_categoria
})


# ------------------------------------------------------
#        10) ID PEDIDO + MONTO
# ------------------------------------------------------

mask_pedido = df["pedido_generado"] == 1
num_pedidos = int(mask_pedido.sum())

pedido_nums = np.random.permutation(np.arange(1, num_pedidos + 1))
id_pedidos = format_id("P", pedido_nums)

df.loc[mask_pedido, "id_pedido"] = id_pedidos
df.loc[~mask_pedido, "id_pedido"] = "NA"

montos = np.zeros(n)
montos[mask_pedido.values] = np.round(
    np.random.uniform(200, 34999, size=num_pedidos), 2
)
df["monto_pedido"] = montos


# ------------------------------------------------------
#        11) EXPORTAR CSV
# ------------------------------------------------------

df = df[[
    "id_gestion",
    "id_cliente",
    "id_asesor",
    "medio",
    "resultado_asesor",
    "fecha_hora",
    "nota",
    "pedido_generado",
    "id_pedido",
    "monto_pedido",
    "producto_categoria"
]]

df.to_csv("gestiones.csv", index=False)
print("Archivo 'gestiones.csv' generado exitosamente.")
print("Pedidos generados:", (df["pedido_generado"] == 1).sum())
print("Porcentaje de pedidos:", (df["pedido_generado"] == 1).mean())
