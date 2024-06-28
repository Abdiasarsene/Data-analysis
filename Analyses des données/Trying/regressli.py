import pandas as pd
from sklearn.linear_model import LinearRegression

# Importez vos données à partir d'un fichier Excel
donnees = pd.read_excel(r"C:\Users\ARMIDE Informatique\Desktop\Projet Académique\BD zone CFA.xlsx")

# Spécifiez les variables indépendantes (X) et dépendante (Y)
X = donnees[['XAF']]
Y = donnees['IDE']

# Créez un modèle de régression linéaire
modele_regression = LinearRegression()

# Entraînez le modèle sur l'ensemble complet de données
modele_regression.fit(X, Y)

# Affichez les coefficients du modèle
print("Coefficient (pente) : ", modele_regression.coef_)
print("Intercept (ordonnée à l'origine) : ", modele_regression.intercept_)
