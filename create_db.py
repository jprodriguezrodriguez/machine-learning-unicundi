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
        "descripcion": "<p>La regresión logística estima la probabilidad de que ocurra un evento, como votar o no votar, en función de un conjunto de datos dado de variables independientes.</p><p>Este tipo de modelo estadístico (también conocido como modelo logit) se utiliza a menudo para la clasificación y el análisis predictivo. Dado que el resultado es una probabilidad, la variable dependiente está acotada entre 0 y 1. En la regresión logística, se aplica una transformación logit sobre las probabilidades, es decir, la probabilidad de éxito dividida por la probabilidad de fracaso.</p>",
        "nombre_de_fuente": "IMB - ¿Qué es la regresión logística?",
        "fuente": "https://www.ibm.com/es-es/topics/logistic-regression",
        "imagen_url": "regresion_logistica.png"
    },
    {
        "nombre": "K-Nearest Neighbors (KNN)",
        "descripcion": "<p>El algoritmo k vecinos más cercanos (KNN) es un clasificador de aprendizaje supervisado no paramétrico que utiliza la proximidad para hacer clasificaciones o predicciones sobre la agrupación de un punto de datos individual. Es uno de los clasificadores de clasificación y regresión más populares y sencillos que se utilizan en machine learning hoy en día.</p> <p>Aunque el algoritmo KNN se puede usar para problemas de regresión o clasificación, se suele utilizar como un algoritmo de clasificación, que parte de la suposición de que se pueden encontrar puntos similares cerca unos de otros.</p>",
        "nombre_de_fuente": "IMB - ¿Qué es el algoritmo KNN?",
        "fuente": "https://es.wikipedia.org/wiki/Algoritmo_de_los_k_vecinos_m%C3%A1s_pr%C3%B3ximos",
        "imagen_url": "https://www.ibm.com/es-es/think/topics/knn"
    },
    {
        "nombre": "Árboles de Decisión",
        "descripcion": "<p>Un árbol de decisión en Machine Learning es una estructura de árbol similar a un diagrama de flujo donde un nodo interno representa una característica (o atributo), la rama representa una regla de decisión y cada nodo hoja representa el resultado.</p><p>El nodo superior en un árbol de decisión en Machine Learning se conoce como el nodo raíz. Aprende a particionar en función del valor del atributo. Divide el árbol de una manera recursiva llamada partición recursiva.</p><p>Esta estructura tipo diagrama de flujo lo ayuda a tomar decisiones. Es una visualización como un diagrama de flujo que imita fácilmente el pensamiento a nivel humano. Es por eso que los árboles de decisión son fáciles de entender e interpretar.</p>",
        "nombre_de_fuente": "Sitiobigdata - Árbol de decisión en Machine Learning",
        "fuente": "https://sitiobigdata.com/2019/12/14/arbol-de-decision-en-machine-learning-parte-1/",
        "imagen_url": "arboles_de_decision.png"
    },
    {
        "nombre": "Random Forest",
        "descripcion": "<p>El bosque aleatorio es un algoritmo de aprendizaje automático de uso común, registrado por Leo Breiman y Adele Cutler, que combina el resultado de múltiples árboles de decisión para llegar a un resultado único. Su facilidad de uso y flexibilidad han impulsado su adopción, ya que maneja problemas de clasificación y regresión.</p><p>El algoritmo de bosque aleatorio es una extensión del método bagging o embolsado, ya que emplea tanto el embolsado como la aleatoriedad de características para crear un bosque de árboles de decisión que no están correlacionados. La aleatoriedad de características, también conocida como embolsado de características o “método aleatorio del subespacio”, genera un subconjunto aleatorio de características, lo que garantiza una baja correlación entre los árboles de decisión. Esta es una diferencia fundamental entre los árboles de decisión y los bosques aleatorios. Mientras que los árboles de decisión consideran todas las posibles divisiones de características, los bosques aleatorios solo seleccionan un subconjunto de esas características.</p>",
        "nombre_de_fuente": "IMB - ¿Qué es el bosque aleatorio?",
        "fuente": "https://www.ibm.com/mx-es/think/topics/random-forest",
        "imagen_url": "random_forest.png"
    },
    {
        "nombre": "Support Vector Machine (SVM)",
        "descripcion": "<p>Una máquina de vectores de soporte (SVM) es un algoritmo de aprendizaje automático supervisado que clasifica los datos al encontrar una línea o hiperplano óptimo que maximice la distancia entre cada clase en un espacio N-dimensional.</p><p>Las SVM fueron desarrolladas en la década de 1990 por Vladimir N. Vapnik y sus colegas, y publicaron este trabajo en un artículo titulado 'Support Vector Method for Function Approximation, Regression Estimation, and Signal Processing' en 1995.</p><p>Las SVM se emplean comúnmente en problemas de clasificación. Distinguen entre dos clases encontrando el hiperplano óptimo que maximiza el margen entre los puntos de datos más cercanos de clases opuestas. El número de características en los datos de entrada determina si el hiperplano es una línea en un espacio bidimensional o un plano en un espacio n-dimensional. Dado que se pueden encontrar múltiples hiperplanos para diferenciar clases, la maximización del margen entre puntos permite al algoritmo encontrar la mejor frontera de decisión entre clases. Esto, a su vez, le permite generalizar bien nuevos datos y hacer predicciones de clasificación precisas. Las líneas adyacentes al hiperplano óptimo se conocen como vectores de soporte, ya que estos vectores atraviesan los puntos de datos que determinan el margen máximo.</p>",
        "nombre_de_fuente": "IMB - ¿Qué son las SVM?",
        "fuente": "https://www.ibm.com/mx-es/think/topics/support-vector-machine",
        "imagen_url": "support_vector_machine.png"
    },
    {
        "nombre": "Gradient Boosting (XGBoost, AdaBoost)",
        "descripcion": "<p>Gradient boosting o potenciación del gradiente, es una técnica de aprendizaje automático utilizado para el análisis de la regresión y para problemas de clasificación estadística, el cual produce un modelo predictivo en forma de un conjunto de modelos de predicción débiles, típicamente árboles de decisión. Construye el modelo de forma escalonada como lo hacen otros métodos de boosting, y los generaliza permitiendo la optimización arbitraria de una función de pérdida diferenciable.</p><p>La idea de la potenciación del gradiente fue originada en la observación realizada por Leo Breiman en donde el Boosting puede ser interpretado como un algoritmo de optimización en una función de coste adecuada. Posteriormente Jerome H. Friedman desarrolló algoritmos de aumento de gradiente de regresión explícita, simultáneamente con la perspectiva más general de potenciación del gradiente funcional de Llew Mason, Jonathan Baxter, Peter Bartlett y Marcus Frean.</p>",
        "nombre_de_fuente": "Wikipedia - Gradient boosting",
        "fuente": "https://es.wikipedia.org/wiki/Gradient_boosting",
        "imagen_url": "gradient_boosting.png"
    },
    {
        "nombre": "Naive Bayes",
        "descripcion": "<p>El clasificador Naive Bayes es un algoritmo de machine learning supervisado que se utiliza para tareas de clasificación como la clasificación de textos. Utiliza principios de probabilidad para realizar tareas de clasificación.</p><p>Naïve Bayes forma parte de una familia de algoritmos de aprendizaje generativo, lo que significa que busca modelar la distribución de las entradas de una clase o categoría determinada. A diferencia de los clasificadores discriminativos, como la regresión logística, no aprende qué características son las más importantes para diferenciar entre clases.</p>",
        "nombre_de_fuente": "IBM - ¿Qué son los clasificadores Naive Bayes?",
        "fuente": "https://www.ibm.com/es-es/think/topics/naive-bayes",
        "imagen_url": "naive_bayes.png"
    }
]

# Convertir a lista de tuplas para insertar correctamente
datos = [(m["nombre"], m["descripcion"], m["fuente"], m["imagen_url"]) for m in modelos]

cursor.executemany('INSERT INTO modelos (nombre, descripcion, fuente, grafico) VALUES (?, ?, ?, ?)', datos)

# Guardar los cambios y cerrar
conn.commit()
conn.close()

print("Base de datos creada y registros insertados.")