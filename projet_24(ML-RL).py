# Importation des données
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Lecture des données
donnees=pd.read_excel(r'Bases de données/DonneSante.xlsx')

# Statistique descriptive
stats=donnees.select_dtypes(exclude=['object']).drop(columns=['Unnamed: 0.2','Unnamed: 0','Unnamed: 0.1']).describe()

# Définition de la base de données sans les catégories et les variables inutiles
dataheath = donnees.select_dtypes(exclude=['object']).drop(columns=['Unnamed: 0.2','Unnamed: 0','Unnamed: 0.1'])

# Visualisation des données manquantes
msno.matrix(dataheath) # Avoir une vue détaillée
msno.bar(dataheath, color='purple') #Avoir une vue globale des données manquantes

# Sélection des variables
y = dataheath['Sexe']  # Variable cible
x = dataheath[['age', 'taux_de_cholesterol', 'taux_de_glycemie', 'pression_arterielle', 'Medicament_code', 'Arthrose_code']]

# Division des données en ensemble d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=808)

# Entraînement du modèle
modele = LogisticRegression()
modele.fit(x_train, y_train)  # Correction : Inverser les arguments y_train et x_train

# Prédiction
y_pred = modele.predict(x_test)

# Évaluation de la performance prédictive
conf_matrix = confusion_matrix(y_test, y_pred)
report_class = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Affichage des résultats
print("Matrice de confusion :\n", conf_matrix)
print("\nRapport de classification :\n", report_class)
print("\nExactitude :\n", accuracy)
