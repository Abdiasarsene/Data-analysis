# Importations des librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
# from linearmodels.tests import hausman
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Importation de la base de données
efm = pd.read_excel(r"c:\Users\ARMIDE Informatique\Downloads\bdd_280.xlsx")

# Préliminaires
print('AFFICHAGE DE LA BASE DEDONNEE')
print('')
print(efm.head())
print(efm.tail())
print("")
print('Information sur la base de données')
print("")
print(efm.info())
print('')
print('Données ménquantes et doublons')
print("")
print(efm.isnull().sum())
print('')
print(efm.duplicated().sum())

# Suppression des certaines variables 
efm_countrycode =efm.drop(columns=["Countrycode", 'JS','PopGrowth','GovBalance','URate'])
print(efm_countrycode)

# Arrondir les variables d'étude
projet_12 = efm_countrycode.round()

# Création d'une nouvelle variable 
projet_12['CAB_EPL'] = projet_12['CAB']*projet_12['EPL']

# Remplisage des données manquantes
projet_dtypes = projet_12.select_dtypes(exclude=['object'])
projet_12 = projet_dtypes.fillna(projet_dtypes.mean())
print(projet_12)

# Importation de la nouvelle base de données pour démarrer l'analyse des données

data_thesis = pd.read_excel(r'projet_12.xlsx')
print(data_thesis.head(), data_thesis.tail())

# Statisitique descrptive
statistic_thesis = data_thesis.drop(columns=['Country', 'Year', 'CAB', 'CAB_EPL']).describe()
print(statistic_thesis)

# Nuage de point afin d'observer la distribution normale
#  Sélection des colones
col1 = data_thesis['UBen'] #Variable expliquée

# Variables explicatives
col2 = data_thesis['SocSpend']
col3 = data_thesis['LFPR']

# Variable de contrôle
col4 = data_thesis['GDPGrowth']
col5 = data_thesis['TradeOpen']
col6 = data_thesis['AgeDep']

# variable modératrice
col7 = data_thesis['EPL']

# Visualisation des données

# Créer une figure et des sous-graphiques
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Nuage de points pour UBen vs. SocSpend
axs[0].scatter(col2, col1, color='blue', alpha=0.5)
axs[0].set_title('UBen vs. SocSpend')
axs[0].set_xlabel('SocSpend')
axs[0].set_ylabel('UBen')

# Nuage de points pour UBen vs. LFPR
axs[1].scatter(col3, col1, color='red', alpha=0.5)
axs[1].set_title('UBen vs. LFPR')
axs[1].set_xlabel('LFPR')
axs[1].set_ylabel('UBen')

# Afficher le graphique
plt.tight_layout()
plt.show()

# Créer une figure et des sous-graphiques
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Nauge de point pour UBen vs. GDPGrowth
axs[0].scatter(col4, col1, color='green', alpha =0.5 )
axs[0].set_title('UBen vs. GDPGrowth')
axs[0].set_xlabel('SocSpend')
axs[0].set_ylabel('UBen')

# Nauge de points pour TradeOpen vs. UBen
axs[1].scatter(col5, col1, color='black', alpha =0.5 )
axs[1].set_title('UBen vs. TradeOpen')
axs[1].set_xlabel('TradeOpen')
axs[1].set_ylabel('UBen')

# Afficher le graphique
plt.tight_layout()
plt.show()

# Matrice de corrélation
correlation = data_thesis.drop(columns = ['Country', 'Year', 'CAB', 'CAB_EPL']).corr()
correlation

# Visualisation graphique de la matrice de corrélatuion
sns.heatmap(correlation, annot=True, cmap = 'coolwarm')
plt.title ('Carte de coorélation')
plt.show()

# Histogramme"
# Configuration de seaborn
sns.set(style="whitegrid")

# Création de l'histogramme avec seaborn
sns.histplot(col1, bins=10, kde=True, color='green')

# Ajout de titres et de labels
plt.title('Distribution de UBen')
plt.xlabel('UBen')
plt.ylabel('Fréquence')

# Affichage de l'histogramme
plt.show()

# Analyse éconoémtrique

# Regression linéaire multiple
y = col1
x = data_thesis[['SocSpend', 'LFPR','GDPGrowth','TradeOpen','AgeDep','EPL']]

# Ajout des constants
x = sm.add_constant(x)

# Création du modèle
model = sm.OLS(y, x).fit()

# Affichage des résultats
print(model.summary())

# tes de diagnostic
residual = model.resid

# Test de Breusch-Pagan
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residual, x)

# Affichage des résultats
print("Test de Breusch")
print('LM Statistic :', lm)
print("LM p-value :", lm_p_value)
print('F-Statistic : ', fvalue)
print('F p-valeu : ', f_p_value)

# Test de Durbin-Watson (Test d'autocorrélation)
durbin_watson_statistic = durbin_watson(residual)

# Afficher le résultat du test
print("Test de Durbin-Watson")
print("Statistique de Durbin-Watson :", durbin_watson_statistic)
print('')

# Test de Shapiro-Wilk (Normalité des résidus)
shapiro_test_statistic, shapiro_test_p_value = shapiro(residual)

# Afficher le résultat du test
print("Test de Shapiro-Wilk")
print("")
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

# Sélection des pays dans la base de données
data = data_thesis['Country'].unique()

# Ajouter des variables indicatrices pour chaque pays
data = pd.get_dummies(data_thesis, columns=['Country'], drop_first=True)

# Supprimer les lignes avec des valeurs manquantes, si nécessaire
data = data.dropna()

# Convertir les colonnes booléennes en numériques (0 pour False et 1 pour True)
data[['Country_Armenia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Belarus', 'Country_Belgium', 'Country_Bosnia and Herzegovina', 'Country_Bulgaria', 'Country_Croatia', 'Country_Cyprus', 'Country_Denmark', 'Country_Estonia', 'Country_Finland', 'Country_France', 'Country_Georgia', 'Country_Germany', 'Country_Greece', 'Country_Hungary', 'Country_Iceland', 'Country_Ireland', 'Country_Italy', 'Country_Latvia', 'Country_Lithuania', 'Country_Luxembourg', 'Country_Malta', 'Country_Moldova', 'Country_Netherlands', 'Country_North Macedonia', 'Country_Norway', 'Country_Poland', 'Country_Portugal', 'Country_Romania', 'Country_Russia', 'Country_Serbia', 'Country_Slovakia', 'Country_Slovenia', 'Country_Spain', 'Country_Sweden', 'Country_Switzerland', 'Country_Turkey', 'Country_Ukraine', 'Country_United Kingdom', 'Country_United States']] = data[['Country_Armenia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Belarus', 'Country_Belgium', 'Country_Bosnia and Herzegovina', 'Country_Bulgaria', 'Country_Croatia', 'Country_Cyprus', 'Country_Denmark', 'Country_Estonia', 'Country_Finland', 'Country_France', 'Country_Georgia', 'Country_Germany', 'Country_Greece', 'Country_Hungary', 'Country_Iceland', 'Country_Ireland', 'Country_Italy', 'Country_Latvia', 'Country_Lithuania', 'Country_Luxembourg', 'Country_Malta', 'Country_Moldova', 'Country_Netherlands', 'Country_North Macedonia', 'Country_Norway', 'Country_Poland', 'Country_Portugal', 'Country_Romania', 'Country_Russia', 'Country_Serbia', 'Country_Slovakia', 'Country_Slovenia', 'Country_Spain', 'Country_Sweden', 'Country_Switzerland', 'Country_Turkey', 'Country_Ukraine', 'Country_United Kingdom', 'Country_United States']].astype(int)

# Séparer les variables indépendantes (X) et dépendante (y)
X = data[['SocSpend', 'LFPR', 'GDPGrowth', 'TradeOpen', 'AgeDep', 'EPL', 'Country_Armenia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Belarus', 'Country_Belgium', 'Country_Bosnia and Herzegovina', 'Country_Bulgaria', 'Country_Croatia', 'Country_Cyprus', 'Country_Denmark', 'Country_Estonia', 'Country_Finland', 'Country_France', 'Country_Georgia', 'Country_Germany', 'Country_Greece', 'Country_Hungary', 'Country_Iceland', 'Country_Ireland', 'Country_Italy', 'Country_Latvia', 'Country_Lithuania', 'Country_Luxembourg', 'Country_Malta', 'Country_Moldova', 'Country_Netherlands', 'Country_North Macedonia', 'Country_Norway', 'Country_Poland', 'Country_Portugal', 'Country_Romania', 'Country_Russia', 'Country_Serbia', 'Country_Slovakia', 'Country_Slovenia', 'Country_Spain', 'Country_Sweden', 'Country_Switzerland', 'Country_Turkey', 'Country_Ukraine', 'Country_United Kingdom', 'Country_United States']]
y = data['UBen']

# Convertir les autres colonnes en types numériques si nécessaire
X = X.astype(float)

# Ajouter une constante à X (constante pour le terme d'interception)
X = sm.add_constant(X)

# Estimer le modèle à effets fixes
model = sm.OLS(y, X).fit()

# Afficher les résultats de la régression
print(model.summary())



# Séparer les variables indépendantes (X) et dépendante (y)
X = data[['SocSpend', 'LFPR', 'GDPGrowth', 'TradeOpen', 'AgeDep', 'EPL','Country_France', 'Country_United States','Country_Belgium','Country_Austria' ]]
y = data['UBen']

# Convertir les autres colonnes en types numériques si nécessaire
X = X.astype(float)

# Ajouter une constante à X (constante pour le terme d'interception)
X = sm.add_constant(X)

# Estimer le modèle à effets fixes
model = sm.OLS(y, X).fit()

# Afficher les résultats de la régression
print(model.summary())