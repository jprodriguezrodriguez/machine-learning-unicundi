import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

class SaludModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        
   #  método generar_datos()
    def generar_datos(self):
        np.random.seed(42)
        n = 100
        
        # Asegúrate de que las generaciones de arrays tengan exactamente 100 elementos
        edad = np.concatenate([
            np.random.randint(20, 80, 70),  # 70 adultos
            np.random.randint(81, 100, 20), # 20 ancianos
            np.random.randint(1, 20, 10)    # 10 jóvenes
        ])
        
        # Verificación crítica
        assert len(edad) == 100, f"Error: Se generaron {len(edad)} edades en lugar de 100"
        
        # Resto de generaciones (asegurar mismo tamaño)
        presion = np.random.normal(120, 15, 100).astype(int)
        colesterol = np.random.normal(200, 30, 100).astype(int)
        sintomas = np.random.choice(["Leves", "Moderados", "Graves"], size=100, p=[0.6, 0.3, 0.1])
        
        # Crear DataFrame verificando longitudes
        self.df = pd.DataFrame({
            "ID": range(1, 101),
            "Edad": edad,
            "Presion": presion,
            "Colesterol": colesterol,
            "Sintomas": sintomas,
            "Enfermedad": ((presion > 140) | (colesterol > 240) | (sintomas == "Graves")).astype(int)
        })
        
        print("\nDEBUG: DataFrame generado correctamente con", len(self.df), "registros")
        return self.df
    
    def entrenar_modelo(self):
        df_encoded = pd.get_dummies(self.df, columns=["Sintomas"])
        X = df_encoded.drop("Enfermedad", axis=1)
        y = df_encoded["Enfermedad"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion": confusion_matrix(y_test, y_pred).tolist()
        }
        
        return self.metrics
    
    def generar_grafico_edad_presion(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gráfico de dispersión
        colors = np.where(self.df["Enfermedad"] == 1, 'red', 'green')
        ax.scatter(
            self.df["Edad"], 
            self.df["Presion"], 
            c=colors,
            alpha=0.6,
            label="Datos reales"
        )
        
        # Líneas de referencia
        ax.axhline(y=140, color='orange', linestyle='--', label='Presión alta')
        ax.axvline(x=80, color='blue', linestyle=':', label='Edad avanzada')
        
        ax.set_xlabel("Edad")
        ax.set_ylabel("Presión Arterial")
        ax.set_title("Relación Edad-Presión y Enfermedad")
        ax.legend()
        ax.grid(True)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def generar_grafico_confusion(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        
        cm = np.array(self.metrics["confusion"])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        ax.set_title("Matriz de Confusión")
        fig.colorbar(im)
        
        classes = ["Sano", "Enfermo"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicción')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def get_datos_para_tabla(self):
        """Devuelve los 100 registros para mostrar en la tabla HTML"""
        print("\nDebug: Mostrando primeros 5 registros del DataFrame:")
        print(self.df.head())
        
        datos = self.df.to_dict(orient='records')
        print("\nDebug: Primer registro convertido a dict:")
        print(datos[0] if datos else "No hay datos")
        
        return datos