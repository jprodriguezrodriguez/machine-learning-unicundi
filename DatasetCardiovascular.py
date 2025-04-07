import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

class SaludModel:
    def __init__(self):
        # Inicializamos el modelo de regresión logística
        self.model = LogisticRegression(max_iter=100, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = None
        
   #  método generar_datos()
    def generar_datos(self):
        np.random.seed(42)
        n = 100
        
        # 1. Generación de variables independientes
        edad = np.concatenate([
            np.random.randint(20, 80, 70),  # 70 adultos
            np.random.randint(81, 100, 20), # 20 ancianos
            np.random.randint(1, 20, 10)    # 10 jóvenes
        ])
        
        # Verificación crítica
        assert len(edad) == 100, f"Error: Se generaron {len(edad)} edades en lugar de 100"
        
        # Variable categórica: Síntomas
        sintomas = np.random.choice(
            ["Leves", "Moderados", "Graves"], 
            size=n, 
            p=[0.6, 0.3, 0.1]
        )
        
        # Variables numéricas continuas
        presion = np.where(
            edad > 80,
            np.random.normal(150, 10, n).astype(int),
            np.where(
                edad < 20,
                np.random.normal(110, 10, n).astype(int),
                np.random.normal(120, 15, n).astype(int)
            )
        )

        # Resto de generaciones (asegurar mismo tamaño)
        presion = np.random.normal(120, 15, 100).astype(int)
        colesterol = np.random.normal(200, 30, 100).astype(int)
        sintomas = np.random.choice(["Leves", "Moderados", "Graves"], size=100, p=[0.6, 0.3, 0.1])
        
        colesterol = np.where(
            edad > 80,
            np.random.normal(260, 30, n).astype(int),
            np.where(
                edad < 20,
                np.random.normal(150, 20, n).astype(int),
                np.random.normal(200, 30, n).astype(int)
            )
        )
        
        # Variable objetivo (dependiente)
        enfermedad = np.where(
            (presion > 140) | (colesterol > 240) | (sintomas == "Graves"),
            1, 0
        )
        
        # Crear DataFrame verificando longitudes
        self.df = pd.DataFrame({
            "ID": range(1, n+1),
            "Edad": edad,
            "Presion": presion,
            "Colesterol": colesterol,
            "Sintomas": sintomas,
            "Enfermedad": ((presion > 140) | (colesterol > 240) | (sintomas == "Graves")).astype(int)
        })
        
        print("\nDEBUG: DataFrame generado correctamente con", len(self.df), "registros")
        return self.df
    
    def preparar_datos(self):
        """Prepara los datos para el modelo"""
        # Convertir variable categórica a dummy variables
        df_encoded = pd.get_dummies(self.df, columns=["Sintomas"], drop_first=True)
        
        # Separar variables independientes (X) y dependiente (y)
        X = df_encoded.drop(["Enfermedad", "ID"], axis=1)
        y = df_encoded["Enfermedad"]
        
        # División train-test (70-30)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Estandarización de variables numéricas
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def entrenar_modelo(self):
        """Entrena el modelo y calcula métricas"""
        # Verificar si los datos están preparados
        if self.X_train is None:
            self.preparar_datos()
        
        # Entrenar modelo
        self.model.fit(self.X_train, self.y_train)
        
        # Realizar predicciones
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calcular métricas
        self.metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc_score(self.y_test, y_prob),
            "confusion": confusion_matrix(self.y_test, y_pred).tolist(),
            "y_pred": y_pred,
            "y_prob": y_prob
        }
        
        return self.metrics
    
    def generar_grafico_edad_presion(self):
        """Genera gráfico de dispersión Edad vs Presión"""
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
        """Genera matriz de confusión"""
        if self.metrics is None:
            self.entrenar_modelo()
            
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
    
    def generar_curva_roc(self):
        """Genera la curva ROC"""
        if self.metrics is None:
            self.entrenar_modelo()
            
        fpr, tpr, _ = roc_curve(self.y_test, self.metrics['y_prob'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def generar_grafico_metricas(self):
        """Gráfico de barras con las métricas principales"""
        if self.metrics is None:
            self.entrenar_modelo()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        metric_values = [
            self.metrics['accuracy'],
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1'],
            self.metrics['roc_auc']
        ]
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        bars = ax.bar(metric_names, metric_values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_title('Métricas del Modelo')
        ax.set_ylabel('Valor')
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
    def generar_grafico_coeficientes(self):
        """Muestra la importancia de las características"""
        if not hasattr(self.model, 'coef_'):
            self.entrenar_modelo()
        
        coef = self.model.coef_[0]
        features = ['Edad', 'Presion', 'Colesterol', 'Sintomas_Moderados', 'Sintomas_Graves']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=coef, y=features, palette='viridis', ax=ax)
        ax.set_title('Importancia de las Características')
        ax.set_xlabel('Coeficiente')
        ax.set_ylabel('Característica')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode('utf-8')
    
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
    
    def generar_todos_graficos(self):
        """Genera todos los gráficos y devuelve en un diccionario"""
        return {
            'edad_presion': self.generar_grafico_edad_presion(),
            'confusion': self.generar_grafico_confusion(),
            'roc': self.generar_curva_roc(),
            'metricas': self.generar_grafico_metricas(),
            'coeficientes': self.generar_grafico_coeficientes()
        }
    
    def run_pipeline(self):
        """Ejecuta todo el proceso completo"""
        print("Generando datos...")
        self.generar_datos()
        
        print("\nPreparando datos...")
        self.preparar_datos()
        
        print("\nVariables independientes utilizadas:")
        print("- Edad (numérica)")
        print("- Presión arterial (numérica)")
        print("- Colesterol (numérica)")
        print("- Síntomas (categórica: Leves/Moderados/Graves)")
        
        print("\nEntrenando modelo...")
        metrics = self.entrenar_modelo()
        
        print("\n=== Métricas del Modelo ===")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1-Score: {metrics['f1']:.2f}")
        print(f"AUC-ROC: {metrics['roc_auc']:.2f}")
        
        return metrics