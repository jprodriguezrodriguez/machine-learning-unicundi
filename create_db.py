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
    ('Regresión Lineal', 'Modelo que predice una variable dependiente usando una relación lineal.', 'https://es.wikipedia.org/wiki/Regresión_lineal', 'regresion_lineal.png'),
    ('Árbol de Decisión', 'Modelo basado en árboles que divide datos en ramas para predecir resultados.', 'https://es.wikipedia.org/wiki/%C3%81rbol_de_decisi%C3%B3n', 'arbol_decision.png')
]

cursor.executemany('INSERT INTO modelos (nombre, descripcion, fuente, grafico) VALUES (?, ?, ?, ?)', modelos)

# Guardar los cambios y cerrar
conn.commit()
conn.close()

print("Base de datos creada y registros insertados.")