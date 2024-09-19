# Importation des bibliothèques
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt

# Importation des bases des données
bd1 = pd.read_excel(r'Bases de données/BD zone CFA.xlsx')
bd2 = pd.read_stata(r'Bases de données/BDD dpae-annuelles-france.dta')

# Concaténation des bases des données
base_de_donne = pd.concat([bd1, bd2], ignore_index =True )

print(base_de_donne.count())

# # Appel des bases de données
# col1 = 'XAF'
# col2 = 'IDE'
# col3 = 'DPAEbrut'
# col4 = 'DPAEcvs'

# # Redéfinition de la base de données
# nouvelle_base_de_donne = pd.DataFrame(base_de_donne, columns = (col1, col2, col3, col4))
# nouvelle_base_de_donne = nouvelle_base_de_donne.fillna(nouvelle_base_de_donne.median())

# # Statistique descriptive, covariance et corrélation
# statistique = nouvelle_base_de_donne.describe()
# correlation = nouvelle_base_de_donne.corr()
# covariance = nouvelle_base_de_donne.cov()

# # Affichage des résultats
# print( "Résultats de la statisitique :" )
# print('')
# print(statistique)
# print('')
# print("Résultats de la corrélation :" )
# print('')
# print(correlation)
# print('')
# print("Résultats de la convariance :")
# print('')
# print(covariance)
# print('')

# # Test d'hypothèses
#     # Définition des colonnes en data
# data1 = nouvelle_base_de_donne[col1]
# data2 = nouvelle_base_de_donne[col2]
# data3 = nouvelle_base_de_donne[col3]

#     # Effectuer le test de khi-deux
# try : 
#     # Tableau de contingence
#     contingency = pd.crosstab(data1, data2)
#     chi2_stat, p_value, dof, expected =stats.chi2_contingency(contingency)

#     # Niveau de signification
#     alpha = 0.05
#     print('Test de khi-deux')

# # Conclusion du test de Khi-deux
#     if p_value < alpha : 
#         print ("Il y a une relation significative entre les deux variables (rejeter l'hypothèse nulle)")
#     else : 
#         print("Il n'y a pas suffisamment de preuves pour rejeter l'hypothèse nulle")
# except :
#     print("Erreur")

# # Regression linéaire
# # Définition des colonnes en data
# x = data1
# y = data2

# try :
#     # Ajout du constant à la variable indépendante
#     x = sm.add_constant(x)

# # Regression linéaire
#     model = sm.OLS(y, x).fit()
#     prediction = model.predict(x)

# # Affichage des résultats :
#     print(' ')
#     print(' RESULTATS DE LA MODELISATION :  ')
#     print(' ')
#     print('Regression linéaire:')
#     print(model.summary())
#     print('')
#     print('Prédiction après modélisation : ')
#     print(prediction)
# except Exception as e: 
#     print('Erreur : ', e)

# try:
# # Visualisation des données
#     plt.plot(x, prediction, color='red', linewidth=2)
#     plt.show()

# # Visualisation des données
#     plt.scatter(x, y, color='black')
#     plt.plot(x, prediction, color='red', linewidth=2)

# # Ajout des titres
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.xticks(())
#     plt.yticks(())
#     plt.title("Regression linéaire")
#     plt.show()
# except Exception as e:
#     print("Erreur", e)