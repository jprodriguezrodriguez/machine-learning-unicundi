import numpy as np

# Datos de producción
produccion = np.array([
    [1000, 1050, 1100, 1150, 1200, 1250, 1300],
    [950, 1000, 1050, 1100, 1150, 1200, 1250],
    [900, 950, 1000, 1050, 1100, 1150, 1200],
    [850, 900, 950, 1000, 1050, 1100, 1150],
    [800, 850, 900, 950, 1000, 1050, 1100],
    [750, 800, 850, 900, 950, 1000, 1050]
])

# 1. Creación de Arrays
print("Matriz de Producción:\n", produccion)

## 2. Manipulación de Arrays
#print("\nTransposición de la matriz:\n", produccion.T)
#print("\nMatriz a un solo vector:", produccion.flatten())
#print("\nConcatenación de la matriz consigo misma:\n", np.concatenate((produccion, produccion), axis=0))
#
## 3. Funciones Matemáticas
#print("\nRaíz cuadrada de los valores:\n", np.sqrt(produccion))
#print("\nSeno de los valores:\n", np.sin(produccion))
#print("\nSuma total de producción:", np.sum(produccion))
#print("\nProducto total de producción:", np.prod(produccion))
#
## 4. Estadísticas
#print("\nMedia de producción:", np.mean(produccion))
#print("\nMediana de producción:", np.median(produccion))
#print("\nDesviación estándar:", np.std(produccion))
#print("\nValor mínimo:", np.min(produccion))
#print("\nValor máximo:", np.max(produccion))
#print("\nÍndice del valor mínimo:", np.argmin(produccion))
#print("\nÍndice del valor máximo:", np.argmax(produccion))

# 5. Álgebra Lineal
#print("\nMatriz identidad del mismo tamaño:\n", np.eye(produccion.shape[0]))
#print("\nDeterminante de una submatriz 2x2:\n", np.linalg.det(produccion[:2, :2]))
#autovalores, autovectores = np.linalg.eig(produccion[:2, :2])
#print("\nAutovalores de la submatriz 2x2:\n", autovalores)
#print("\nAutovectores de la submatriz 2x2:\n", autovectores)

# 6. Funciones de Indexación y Selección
#print("\nProducción del tercer mes de todas las fábricas:\n", produccion[:, 2])
#print("\nProducción de la segunda fábrica en todos los meses:\n", produccion[1, :])
#print("\nSubmatriz con las primeras 3 fábricas y 3 primeros meses:\n", produccion[:3, :3])
#print("\nValores mayores a 1000:\n", produccion[produccion > 1000])