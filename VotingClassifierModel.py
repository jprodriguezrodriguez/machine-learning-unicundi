import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import io
import os

class VotingClassifierModel:
    def __init__(self, filepath):
        # Detectar extensión
        ext = os.path.splitext(filepath)[1].lower()

        # Leer datos según tipo de archivo
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            raise ValueError(f"Formato no soportado: {ext}")

        # Seleccionar variables
        self.X = df[['Edad', 'Presion_Arterial_Reposo', 'Colesterol']]
        y = df['Diagnostico_Enfermedad_Cardiaca']
        self.y = (y > 0).astype(int)

        # Inicializar atributos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        
    def splitDataset(self, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )

    def buildVotingModel(self):
        clf1 = LogisticRegression(max_iter=1000, random_state=42)
        clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf3 = SVC(probability=True, random_state=42)
        self.model = VotingClassifier(
            estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
            voting='soft'
        )

    def trainModel(self):
        if self.model is None:
            self.buildVotingModel()
        self.model.fit(self.X_train, self.y_train)

    def evaluateModel(self):
        y_pred = self.model.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_test, y_pred))

    def saveModelJoblib(self, path='models/voting_model_heart.joblib'):
        joblib.dump(self.model, path)

    def loadTrainedModel(self, path='models/voting_model_heart.joblib'):
        self.model = joblib.load(path)
        return self.model