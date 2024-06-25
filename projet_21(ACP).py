# Importation des librairies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# # Importation des bases de données 
# Première base de données
# courses = pd.read_csv(r"c:\Users\ARMIDE Informatique\Downloads\courses_info.csv")
# print(courses.info())

# # Deuxième base de données
# my_course = pd.read_csv(r"c:\Users\ARMIDE Informatique\Downloads\my_courses.csv")
# print('')
# print(my_course.info())

# # Troisième base de données
# mystery = pd.read_csv(r"c:\Users\ARMIDE Informatique\Downloads\mystery.csv")
# print('')
# print(mystery.isnull().sum())
# print('')
# print(mystery.duplicated().sum())
# print('')

# 

# Charger les données depuis un fichier CSV par exemple
data = pd.read_excel(r"Bases de données/bdd_diif-in-diff.xlsx")

# Séparer les caractéristiques (features) des étiquettes (labels) si nécessaire
X = data.drop(columns=["CO2eq"])
y = data["CO2eq"]

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Créer un objet PCA et ajuster les données
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Tracer le graphique de la variance expliquée par chaque composante principale
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.xlabel('Nombre de Composantes Principales')
plt.ylabel('Variance Expliquée')
plt.title('Variance Expliquée par les Composantes Principales')
plt.grid(True)
plt.show()

# Choisir le nombre de composantes principales à utiliser en fonction du graphique ci-dessus
# par exemple, supposons que vous souhaitez conserver les deux premières composantes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Vous pouvez maintenant utiliser X_pca pour l'analyse ultérieure
