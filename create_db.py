import sqlite3

# Crear (o abrir si ya existe) la base de datos
conn = sqlite3.connect('modelos.db')

# Crear un cursor para ejecutar comandos SQL
cursor = conn.cursor()

# Crear la tabla
cursor.execute('''
    CREATE TABLE IF NOT EXISTS modelos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        descripcion TEXT NOT NULL,
        fuente TEXT NOT NULL,
        grafico TEXT
    )
''')

# Insertar registros de ejemplo
modelos = [
    {
        "nombre": "Regresión Logística",
        "descripcion": "Modelo de clasificación lineal que estima la probabilidad de pertenencia a una clase usando la función sigmoide.",
        "fuente": "https://es.wikipedia.org/wiki/Regresi%C3%B3n_log%C3%ADstica",
        "imagen_url": "https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg"
    },
    {
        "nombre": "K-Nearest Neighbors (KNN)",
        "descripcion": "Algoritmo de clasificación que asigna una clase a un punto nuevo en función de la mayoría de los k vecinos más cercanos.",
        "fuente": "https://es.wikipedia.org/wiki/Algoritmo_de_los_k_vecinos_m%C3%A1s_pr%C3%B3ximos",
        "imagen_url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/KnnClassification.svg"
    },
    {
        "nombre": "Árboles de Decisión",
        "descripcion": "Modelo predictivo que divide el espacio de características en segmentos mediante decisiones binarias representadas en forma de árbol.",
        "fuente": "https://es.wikipedia.org/wiki/%C3%81rbol_de_decisi%C3%B3n",
        "imagen_url": "https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png"
    },
    {
        "nombre": "Random Forest",
        "descripcion": "Conjunto de árboles de decisión entrenados sobre subconjuntos aleatorios del conjunto de entrenamiento; utiliza votación para clasificación.",
        "fuente": "https://es.wikipedia.org/wiki/Bosque_aleatorio",
        "imagen_url": "https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png"
    },
    {
        "nombre": "Support Vector Machine (SVM)",
        "descripcion": "Algoritmo que encuentra el hiperplano óptimo para separar clases maximizando el margen entre ellas.",
        "fuente": "https://es.wikipedia.org/wiki/M%C3%A1quinas_de_vectores_de_soporte",
        "imagen_url": "https://upload.wikimedia.org/wikipedia/commons/2/20/SVM_margin.png"
    },
    {
        "nombre": "Gradient Boosting (XGBoost, AdaBoost)",
        "descripcion": "Técnica de ensamblado que combina múltiples modelos débiles (como árboles) en uno más fuerte, usando aprendizaje secuencial.",
        "fuente": "https://en.wikipedia.org/wiki/Gradient_boosting",
        "imagen_url": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Boosting_Figure.png"
    },
    {
        "nombre": "Naive Bayes",
        "descripcion": "Clasificador probabilístico basado en el Teorema de Bayes con la suposición de independencia entre características.",
        "fuente": "https://es.wikipedia.org/wiki/Clasificador_naive_Bayes",
        "imagen_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Bayes%27_Theorem_MMB_01.jpg/800px-Bayes%27_Theorem_MMB_01.jpg"
    }
]

# Convertir a lista de tuplas para insertar correctamente
datos = [(m["nombre"], m["descripcion"], m["fuente"], m["imagen_url"]) for m in modelos]

cursor.executemany('INSERT INTO modelos (nombre, descripcion, fuente, grafico) VALUES (?, ?, ?, ?)', datos)

# Guardar los cambios y cerrar
conn.commit()
conn.close()

print("Base de datos creada y registros insertados.")