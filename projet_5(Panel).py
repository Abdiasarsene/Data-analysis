# Importation des librairies
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Importation de la base de données
data_GES = pd.read_excel(r'Bases de données/bdd_diif-in-diff.xlsx')

print('')
print('AFFICHAGES DES RESULTATS DE L\'ETUDE')
print("")

# Préliminaires
print('Affichage de la base de données')
print('')
print(data_GES.head())
print(data_GES.tail())
print('')
print(data_GES.columns)
print('')
print(data_GES.info())
print('')
print('Doublons et données manquantes')
print(data_GES.isnull().sum())
print(data_GES.duplicated().sum())
print('')

# Statistique descriptive 
stat = data_GES.drop(columns=['Year','Traitement ','Apres_intervention' ]).describe()
print('STATISTIQUE DESCRIPTIVE')
print('')
print(stat)
print("")

print('MODELISATION')
print('')
print('Deuxième modèle économétrique : Regression linéaire multiple')
print('')

# Sélections des variables d'études
y = data_GES['CO2eq'] #variable dépendante
x = data_GES[['Prix_carbone', 'invest_energie', 'populations', 'PIB']] #variables indêpendantes

# Ajout du constant aux variables explicatives
x = sm.add_constant(x)

# Estimation du modèle
model = sm.OLS(y,x).fit()

# Affichage du modèle 
print(model.summary())
# Les test de diagnostics après estimation du modèle

print('')
print('TEST DE DIAGNOSTIC')
print('')

# Calculer les résidus
residuals = model.resid

# Test de Breusch-Pagan (Test d'homoscédasticité)
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuals, x)

# Afficher les résultats du test
print("Test de Breusch-Pagan")
print('')
print("LM Statistic :", lm)
print("LM p-value :", lm_p_value)
print("F-Statistic :", fvalue)
print("F p-value :", f_p_value)
print('')

# Test de Durbin-Watson (Test d'autocorrélation)
durbin_watson_statistic = durbin_watson(residuals)

# Afficher le résultat du test
print("Test de Durbin-Watson")
print("Statistique de Durbin-Watson :", durbin_watson_statistic)
print('')

# Test de Shapiro-Wilk (Normalité des résidus)
shapiro_test_statistic, shapiro_test_p_value = shapiro(residuals)

# Afficher le résultat du test
print("Test de Shapiro-Wilk")
print("Statistique de test :", shapiro_test_statistic)
print("p-value :", shapiro_test_p_value)
print('')

# Test de multicolinéarité
# Ajouter une colonne pour la constante
x = add_constant(x)

# Calculer les valeurs VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

# Afficher les valeurs VIF
print('Test de multicolinéarité')
print('')
print("Valeurs VIF :")
print(vif_data)

print('')
print('VISUALISATION DES DONNEES')
print('')

# Visualisation graphique
# Visualisation de la variable CO2eq
print('Visualisation de la variable CO2eq')
print('')

y = data_GES['CO2eq'] #variable dépendante
x = data_GES[['Prix_carbone', 'invest_energie', 'populations', 'PIB']] #variables indêpendantes

fig, axis = plt.subplots(2,2, figsize=(12,10))

axis[0,0].scatter(y, data_GES['Prix_carbone'], color ='purple', alpha=0.5)
axis[0,0].set_title('CO2eq vs Prix du carbone')
axis[0,0].set_xlabel('Prix du carbone')
axis[0,0].set_ylabel('CO2eq')

axis[0,1].scatter(y, data_GES['invest_energie'], color='red', alpha =0.5)
axis[0,1].set_title('CO2eq vs Investissement de l\'énergie renouvelable')
axis[0,1].set_xlabel('Investissement de l\'énergie renouvelable')
axis[0,1].set_ylabel('CO2eq')

# fig, axis = plt.subplots(1,2, figsize = (12, 5))
axis[1,0].scatter(y, data_GES['PIB'], color='black', alpha=0.5)
axis[1,0].set_title('CO2eq vs PIB')
axis[1,0].set_ylabel('CO2eq')
axis[1,0].set_xlabel('PIB')

axis[1,1].scatter(y, data_GES['populations'],color='green', alpha=0.5)
axis[1,1].set_title('CO2eq vs populations')
axis[1,1].set_ylabel('CO2eq')
axis[1,1].set_xlabel('populations')

# Affichage des résultats
plt.tight_layout()
plt.show()

# Histogramme
# Configuration de seaborn
sns.set(style="whitegrid")

# Création de l'histogramme avec seaborn
sns.histplot(y, bins=10, kde=True, color='purple')

# Ajout de titres et de labels
plt.title('Distribution de UBen')
plt.xlabel('UBen')
plt.ylabel('Fréquence')

# Affichage de l'histogramme
plt.show()