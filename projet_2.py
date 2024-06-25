# Importation des librairies
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# Importations des données
dataset1 = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\BDD-NOUVEAU 1.xlsx')
datasets1 = dataset1.dropna(axis=0)
dataset2 = pd.read_stata(r'c:\Users\ARMIDE Informatique\Desktop\Formation pratique\exemple touirisme.dta')
datasets2 = dataset2.dropna(axis=0)
dataset3 = pd.read_spss(r'c:\Users\ARMIDE Informatique\Desktop\Formation pratique\Data Exo.sav')
datasets3 = dataset3.dropna(axis=0)

# Liaison des bases de données
liaison_datas = pd.concat([datasets1, datasets2, datasets3], ignore_index= True)

# Selection des variables d'études
col1 = 'Sexe'
col2 = 'Recetteglobale'
col3 = 'Part de marché immobilier en %'
col4 = 'Nombredetouristeenmilliers'
col6 = 'Dépensesdestouristesenmilli'

# Création d'une nouvelle base de données
nouvelle_dataset = pd.DataFrame(liaison_datas, columns = (  col2, col3, col4, col6))
nouvelle_datasets = pd.DataFrame(liaison_datas, columns = (  col2, col3, col4, col6))

# Identification des données manquantes
db = nouvelle_dataset.isnull().sum()
bd = np.isinf(nouvelle_dataset).sum()
print(db, bd)

# Remplissage des données manquantes
nouvelle_dataset= nouvelle_dataset.fillna(nouvelle_dataset[col4].median)
nombre_dobs = 30
nouvelle_dataset = nouvelle_dataset.head(nombre_dobs)

# Supprimer les lignes contenant des valeurs infinies
# nouvelle_dataset = nouvelle_dataset[~np.isinf(nouvelle_dataset).any(axis=1)]


# Pré-traitement des données
base_de_donne = nouvelle_dataset
base_de_donnes = nouvelle_datasets


# Statistique, covariance, corrélation
stat = base_de_donne.describe()
cor = base_de_donne.corr()
cova = base_de_donne.cov()
print(stat, cor, cova)

# Régression linéaire et logistique
    # selection des variables 
X = var_dep = base_de_donne[col4]
Y =  var_indp = base_de_donne[col4]

#     # Encodage de ma base de donnée
# x = base_de_donnes[col2]
# y =  var_indp = base_de_donnes[[col3, col5]]


# Ajout de la var dep
X = sm.add_constant(X)
# x = sm.add_constant(x)

# Régression linéaire
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

# # Régression logistique
# logit_model = sm.Logit(y, x).fit()
# predictions_logit = logit_model.predict(X)

# Affichage des résultats
print(model.summary(), predictions )
# print(logit_model.summary(), predictions_logit)

# # Visualisation
# # Graphique de la corrélation
# sns.heatmap(cor, annot=True)
# plt.show()

# # Graphique de la covariance

# sns.heatmap(cova, annot=True)
# plt.show()

# # Graphique de la statistique
# sns.heatmap(stat, annot=True)
# plt.show()