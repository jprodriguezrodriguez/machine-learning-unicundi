from flask import Flask, render_template, request, g
import linearRegressionML as lr
from DatasetCardiovascular import SaludModel
import sqlite3

app = Flask(__name__)
modelo = SaludModel()

DATABASE = 'modelos.db'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diferencias-ia-ml")
def IaMlDifferences():
    return render_template("diferencias-ia-ml.html")

@app.route('/casos-de-exito-ml')
def successCases():
    casesList = [
        {
            'title': 'Diagnóstico Médico Asistido por IA',
            'sector': 'Salud',
            'problem': 'Desafíos en la detección temprana y precisa de enfermedades complejas.',
            'algorithm': 'Redes Neuronales Artificiales (ANN)',
            'benefits': 'Mejora en la precisión diagnóstica y detección temprana de enfermedades.',
            'company': 'IBM Watson analiza datos médicos para asistir en diagnósticos más precisos y rápidos.',
            'reference': 'Imagar. (2024). Machine Learning: casos de éxito en diferentes sectores. Recuperado de https://www.imagar.com/informatica/machine-learning-casos-de-exito-en-diferentes-sectores/',
            'referenceUrl': 'https://www.imagar.com/informatica/machine-learning-casos-de-exito-en-diferentes-sectores/'
        },
        {
            'title': 'Sistemas de Recomendación en Plataformas de Streaming',
            'sector': 'Entretenimiento',
            'problem': 'Necesidad de personalizar la experiencia del usuario mediante recomendaciones de contenido.',
            'algorithm': 'Sistemas de Recomendación basados en Filtrado Colaborativo',
            'benefits': 'Incremento en el tiempo de visualización y lealtad del cliente.',
            'company': 'Netflix utiliza algoritmos de Machine Learning para ofrecer recomendaciones basadas en el comportamiento del usuario.',
            'reference': 'Beservices. (2020). Ejemplos de Machine Learning en empresas. Recuperado de https://blog.beservices.es/blog/ejemplos-de-machine-learning-en-empresas',
            'referenceUrl': 'https://blog.beservices.es/blog/ejemplos-de-machine-learning-en-empresas'
        },
        {
            'title': 'Optimización de Rutas en Servicios de Logística',
            'sector': 'Logística y Transporte',
            'problem': 'Minimizar tiempos de entrega y costos operativos en rutas de transporte.',
            'algorithm': 'Algoritmos de Optimización y Aprendizaje Supervisado',
            'benefits': 'Reducción del consumo de combustible y mejora en los tiempos de entrega.',
            'company': 'UPS utiliza Machine Learning para programar rutas que minimizan los giros a la izquierda, optimizando el tiempo de reparto.',
            'reference': 'Beservices. (2020). Ejemplos de Machine Learning en empresas. Recuperado de https://blog.beservices.es/blog/ejemplos-de-machine-learning-en-empresas',
            'referenceUrl': 'https://blog.beservices.es/blog/ejemplos-de-machine-learning-en-empresas'
        }
    ]
    return render_template('casos-de-exito-ml.html', cases=casesList)

@app.route("/regresion-lineal", methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    plot_url = None
    hours = None

    if request.method == "POST":
        try:
            hours = float(request.form["Hours"])
            calculateResult = lr.calculateGrade(hours)
            plot_url = lr.generate_plot(hours)  # Generar gráfica solo si el input es válido
        except ValueError:
            calculateResult = "Invalid input. Please enter a number."
            plot_url = None  # Evita mostrar una gráfica si hay error en la entrada
    
    return render_template("regresion-lineal-prediccion-notas.html", result=calculateResult, plot_url=plot_url, hours=hours)

@app.route("/mapa-mental")
def mindMap():
    return render_template("mapa-mental-regresion-logistica.html")

@app.route('/regresion-logistica')
def LogisticRegression():
    try:
        # 1. Generar datos
        df = modelo.generar_datos()
        if df.empty:
            raise ValueError("El DataFrame generado está vacío")
        
        # 2. Entrenar modelo
        metricas = modelo.entrenar_modelo()
        
        # 3. Obtener datos para tabla (VERIFICACIÓN CRÍTICA)
        tabla_datos = modelo.get_datos_para_tabla()
        if not tabla_datos:
            raise ValueError("No se obtuvieron datos para la tabla")
        
        print("\nDEBUG: Datos para tabla (primer registro):", tabla_datos[0])
        
        # Renderizar plantilla con todos los datos
        return render_template(
            'regresion-logistica-ml.html',
            stats={
                "total": len(df),
                "enfermos": int(df["Enfermedad"].sum()),
                "sanos": len(df) - int(df["Enfermedad"].sum()),
                "edad_promedio": round(df["Edad"].mean(), 1),
                "presion_promedio": round(df["Presion"].mean(), 1),
                "colesterol_promedio": round(df["Colesterol"].mean(), 1),
                "accuracy": round(metricas["accuracy"] * 100, 1)
            },
            grafico_edad_presion=modelo.generar_grafico_edad_presion(),
            grafico_confusion=modelo.generar_grafico_confusion(),
            confusion_matrix=metricas["confusion"],
            tabla_datos=tabla_datos
        )
        
    except Exception as e:
        print(f"\nERROR CRÍTICO: {str(e)}")
        return f"Error al generar datos: {str(e)}", 500

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/modelo-supervisado/<int:modelo_id>')
def mostrar_modelo(modelo_id):
    cur = get_db().cursor()
    cur.execute("SELECT * FROM modelos WHERE id = ?", (modelo_id,))
    modelo = cur.fetchone()
    return render_template('modelos-supervisados.html', modelo=modelo)

if __name__ == "__main__":
    app.run(debug=True)
