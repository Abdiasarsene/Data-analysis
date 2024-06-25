# Importation des librairies
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Analyse statistique et économétrique
print('')
print('Analyse statistique et économétrique')
print("")

    # Importation des bases de données
jeuddonne1 = pd.read_csv(r'Bases de données/BDD evenements.csv')
jeuddonne2 = pd.read_csv(r'Bases de données/BDD operations.csv')

    # Concaténation des bases de données
concat_jeuddonne = pd.concat([jeuddonne1, jeuddonne2], ignore_index= True)

    # Selection des variables d'études
        # Définition des variable économiques
col1 = 'nb_places'
col2 = 'nb_elec'
col6 = 'freq_departs'
col7 = 'freq_arrivees'

nouvveua_jeuddonne = pd.DataFrame(concat_jeuddonne, columns =( col1, col2, col6, col7))

    # Remplissage des données manquantes
filna_dataset = nouvveua_jeuddonne.fillna( nouvveua_jeuddonne[col6].median)

    # Section de la base de données
nbre_dobservation = 200
new_jeuddonne = filna_dataset.head(nbre_dobservation)
print(new_jeuddonne.count())

print('BASE DE DONNEE PRETE POUR LES ANALYSES')
print('')
print(new_jeuddonne.head())
print("")

    # Statistique descriptive, corrélation et covariance
print('ANALYSE STATISTIQUE :')
print("")
print('A- Statistique descriptive, corrélation et covariance')
print("")

stat_jeuddonne = new_jeuddonne.describe()
print(stat_jeuddonne)
print("")

cor_jeuddonne = new_jeuddonne.corr()
print("La matrice de corrélation est :")
print(cor_jeuddonne)
print("")

cov_jeuddonne = new_jeuddonne.cov()
print('La covariance')
print(cov_jeuddonne)
print("")

print('B- Test d\'hyppthèses (Test de Khi-deux)')
print('')
    # Test d'hyppthèses (Khi-deux)
data1 = new_jeuddonne[col1]
data2 = new_jeuddonne[col2]
data6 = new_jeuddonne[col6]

    # Tableau de contingence
table_contingency = pd.crosstab(data1, data2)
chi2_stat, p_value, dof, expected = stats.chi2_contingency(table_contingency)

    # Niveau de significativité
alpha = 0.1

    # Conclusion du test d'hypothèse
if p_value < alpha :
    print('Test de Khi-deux')
    print("")
    print("Il y a une relation significative entre les deux variables (rejeter l'hypothèse nulle)")
else : 
    print("")
    print("Il n'y a pas suffisamment de preuves pour rejeter l'hypothèse nulle")

print('ANALYSE ECONOMETRIQUE')
print('')
    # Anlyse économétrique : 
        # Selection des variables économiques
x = data1
y = data6

# En supposant que 'endog' et 'exog' sont des Series ou des DataFrames pandas
X = pd.to_numeric(x, errors = 'coerce')
Y = pd.to_numeric(y, errors = 'coerce')

    # Ajout de la variable explicative
X = sm.add_constant(X)

    # Création du modèle linéaire
model = sm.OLS(Y, X).fit()

    # Affichage des résultats
print('Résultats de la regression linéaire :')
print("")
print(model.summary())

    # Intervalle de confiance
print("Intervalle de confiance : ")
print("")
print(model.conf_int())
print('')
print('Regression logistique')

# Régression logistique
try:
    logit_model = sm.Logit(Y, X).fit()
    predictions_logit = logit_model.predict(X)
    print(logit_model.summary(), predictions_logit)
except:
    print("Erreur")

# Visualisation des données
# Graphique de la régression linéaire
plt.scatter(X.iloc[:, 1], Y, color = 'red')
plt.plot(X.iloc[:, 1], model.predict(X), color = 'blue')
plt.title('Régression linéaire')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()