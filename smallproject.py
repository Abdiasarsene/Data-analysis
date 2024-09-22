# Importation des bibliothèques
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

# Importation de la base de données
canada = pd.read_excel(r"d:\Projet Informatique\Bases de données\Base Crowdfunding.xlsx")
canada.tail() #Affichage des 05 dernières lignes
canadl = canada.select_dtypes(exclude=['object'])

# Aanalyses exploratoires des données
    # Détection et supression des données manquantes
missing_data = canadl[canadl.duplicated()]
if not missing_data.empty:
    print('La base de données contient des données manquantes')
    msno.bar(canadl)
    canadl = canadl.drop_duplicates()
else:
    print("La base de données est sans danger", canadl)

# Détection des valeurs aberrantes
Q1 = canadl.quantile(0.25)
Q3 = canadl.quantile(0.75)
IQR = Q3 - Q1
outlier = ((canadl < (Q1 - 1.5 * IQR)) | (canadl > (Q3 + 1.5 * IQR)))
valeurs_aberrantes = canadl[outlier.any(axis=1)] 

if not valeurs_aberrantes.empty:
    print('Datasets contains outliers')
    valeurs_aberrantes
else:
    print('Dataset is cleaned, no danger', canadl)

# Statistique descriptive
statistic_canada = canadl.describe()
statistic_canada #résultat de la statistique

    # Matrice de corrélation
corr_canada = canadl.corr()
sns.heatmap(corr_canada, annot=True, cmap='coolwarm')

    # Visualisation des variables 
sns.pairplot(canadl)

    # Visualisation des relations entre la cible et les prédicteurs
    