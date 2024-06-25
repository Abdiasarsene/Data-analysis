# Importation des librairies  
import pandas as pd
import statsmodels.api as sm

# Importatuion des données
stata = pd.read_stata(r'c:\Users\ARMIDE Informatique\Desktop\Projet Académique\Projet personnel\Mémoire Abdias-Rodrigue\fff.dta')
print(stata)

# Variable dépendante
col1 = 'Taux_croissanceco'
y = stata[col1]

# Variables indépendantes
col2 = 'Depense_tourist'
x1 = stata[col2]

col3 = 'Nbre_touriste'
x2 = stata[col3]

col4 = 'Taux_change'
x3 = stata[col4]

# Ajout des constantes aux variables indépendantes
x = sm.add_constant(pd.concat([x1, x2, x3], axis=1))

# Création du modèle
model = sm.OLS(y, x)

# Ajustement du modèle
results = model.fit()

# Affichage des résultats
print(results.summary())
