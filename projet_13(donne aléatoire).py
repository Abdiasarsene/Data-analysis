# Importation des données
import pandas as pd
import numpy as np

# Définir le nombre de lignes à générer
n = 100

# Générer des données aléatoires
data = {
    "Arthrose": np.random.choice(['oui', 'non'], n),
    "age": np.random.randint(20, 91, n),
    "taux_de_cholesterol": np.random.randint(150, 301, n),
    "taux_de_glycemie": np.random.randint(70, 151, n),
    "Medicament": np.random.choice(['Doliprane', 'Simvastatine', 'Rosuvastatine'], n),
    "pression_arterielle": np.random.randint(90, 181, n)
}

# Créer un DataFrame
df = pd.DataFrame(data, columns=(("Arthrose","age","taux_de_cholesterol","taux_de_glycemie","Medicament","pression_arterielle")))

# Creation d'une nouvelle variable par choix aléatoire
m = 100
donne = {"Sexe": np.random.choice(['Masculin', 'Feminin'],m)}
dat = pd.DataFrame(donne,columns=['Sexe'])
dat['Sexe'], codes = pd.factorize(dat['Sexe'])

# Fusion des deux bases de données
base1 = pd.read_excel('donnees_sante.xlsx')
base2= pd.read_excel('dat.xlsx')
base3 = pd.read_excel('sexe.xlsx')
DonneSante = pd.concat([base1, base2], axis=1)

# Supposons que vous ayez déjà votre objet DataFrame DonneSante

# Colonnes à coder
colonnes_a_coder = ['Medicament', 'Arthrose']

# Dictionnaire pour stocker les correspondances entre les catégories et les codes
codes = {}

# Boucle sur chaque colonne à coder
for colonne in colonnes_a_coder:
    # Appliquer pd.factorize() pour coder la colonne et récupérer les codes ainsi que les correspondances
    DonneSante[colonne + '_code'], codes[colonne] = pd.factorize(DonneSante[colonne])

# Renommer une colonne
DonneSante.rename(columns={'Sexe': 'Sexe'}, inplace=True)
DonneSante.drop(columns=['Unnamed: 0'])

datahealth = pd.concat([DonneSante, base3], axis=1)

import numpy as np
import pandas as pd

# Définir le nombre d'observations
n_obs = 200

# Générer des prédicteurs aléatoires
np.random.seed(808)
age = np.random.randint(20, 71, n_obs)
bmi = np.random.uniform(15, 35, n_obs)
blood_pressure = np.random.uniform(90, 180, n_obs)
cholesterol = np.random.uniform(150, 300, n_obs)
glucose = np.random.uniform(70, 180, n_obs)
smoking_status = np.random.choice([0, 1], n_obs)

# Créer une DataFrame
data = pd.DataFrame({
    'Age': age,
    'BMI': bmi,
    'BloodPressure': blood_pressure,
    'Cholesterol': cholesterol,
    'Glucose': glucose,
    'SmokingStatus': smoking_status
})

# Générer la variable cible avec une relation logique
# On utilise une combinaison pondérée des prédicteurs pour déterminer le statut de la maladie
logit = (-0.1 * age + 0.3 * bmi + 0.02 * blood_pressure + 0.02 * cholesterol + 0.05 * glucose + 0.6 * smoking_status)
prob = 1 / (1 + np.exp(-logit))  # Transformation sigmoïde pour obtenir des probabilités
disease_status = (prob > 0.5).astype(int)  # Seuil de 0.5 pour déterminer la présence de la maladie

data['DiseaseStatus'] = disease_status

# Afficher un aperçu des données
print(data.head())

# Sauvegarder le DataFrame dans un fichier CSV
data.to_csv('synthetic_classification_data.csv', index=False)

import numpy as np
import pandas as pd

# Définir le nombre d'observations
n_obs = 500

# Générer des variables aléatoires pour les nouvelles caractéristiques
np.random.seed(808)
age = np.random.randint(20, 80, n_obs)
gender = np.random.choice([0, 1], n_obs)
income = np.random.normal(50000, 15000, n_obs)
education = np.random.choice([0, 1, 2], n_obs)
health_score = np.random.uniform(0, 100, n_obs)
medication = np.random.choice([0, 1], n_obs)

# Créer une DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Income': income,
    'Education': education,
    'HealthScore': health_score,
    'Medication': medication
})

# Introduire des valeurs manquantes de manière aléatoire
nan_indices = np.random.choice(data.index, size=100, replace=False)
data.loc[nan_indices, 'Age'] = np.nan
nan_indices = np.random.choice(data.index, size=80, replace=False)
data.loc[nan_indices, 'Income'] = np.nan
nan_indices = np.random.choice(data.index, size=120, replace=False)
data.loc[nan_indices, 'HealthScore'] = np.nan

# Introduire des valeurs aberrantes
outlier_indices = np.random.choice(data.index, size=10, replace=False)
data.loc[outlier_indices, 'Education'] = 99  # Valeur aberrante pour l'éducation

# Afficher un aperçu des données
print(data.head())

# Sauvegarder le DataFrame dans un fichier CSV
data.to_csv('panel_data_concrete_names.csv', index=False)
