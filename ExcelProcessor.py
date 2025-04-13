import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ExcelProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self.load_file(filepath)
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()

    def detect_delimiter(filepath):
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            dialect = csv.Sniffer().sniff(file.read(1024))
            file.seek(0)
            return dialect.delimiter

    def load_file(self, path):
        if path.endswith('.csv'):
            delimiter = detect_delimiter(path)
            return pd.read_csv(path, sep=delimiter)
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            return pd.read_excel(path)
        else:
            raise ValueError("Archivo no compatible. Usa .csv o .xlsx")


    def preprocess(self):
        if 'Enfermedad' not in self.df.columns:
            raise ValueError("El archivo debe contener la columna 'Enfermedad' como variable objetivo")

        X = self.df.drop(columns='Enfermedad')
        y = self.df['Enfermedad']

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_and_predict(self):
        X_train, X_test, y_train, y_test = self.preprocess()
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        # Combinar resultados
        test_df = pd.DataFrame(X_test, columns=self.df.drop(columns='Enfermedad').columns)
        test_df['Real'] = y_test.values
        test_df['Prediccion'] = predictions

        # Métricas
        metrics = {
            'Accuracy': accuracy_score(y_test, predictions),
            'Precision (macro)': precision_score(y_test, predictions, average='macro'),
            'Recall (macro)': recall_score(y_test, predictions, average='macro'),
        }
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor'])

        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, predictions)
        conf_df = pd.DataFrame(conf_matrix)

        # Gráfica
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matriz de Confusión")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        plt.close(fig)
        image_stream.seek(0)

        return test_df, metrics_df, conf_df, image_stream
