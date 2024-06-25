# Importation des librairies
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

# IMportation de la base de données
data_thesis = pd.read_excel(r'projet_12.xlsx')
print(data_thesis.columns)

# Choix du nombre d'observation
NombreObservation = 891
new_thesis = data_thesis.head(NombreObservation)

# Suppression des variables inutiles
new_thesis = new_thesis.drop(columns =['GDPGrowth','TradeOpen'])

# Visualisez les données des variables uniques
values_data = new_thesis['Country'].unique()

# Vérification des doublons et des données manquantes
value_null = new_thesis.isnull().sum()
doublons = new_thesis.duplicated().sum()

# Statistique descriptive
statistic = new_thesis.drop(columns = ['Year']).describe()

# Histogramme
new_thesis.drop(columns = ['Year']).hist()

# Boxplot
new_thesis.drop(columns = ['Year']).boxplot()

# Regression linéaire multiple

# Sélection des variables
y = new_thesis['UBen'] #variable dépendante
x = new_thesis[['SocSpend','LFPR','EPL','CAB','CAB_EPL']] # Variables indépendantes

# Ajout des constants aux variables indépendantes
x = sm.add_constant(x)

# Construction du modèle
model = sm.OLS(y,x).fit()

# Affichage des résultats
model.summary()

# Prédiction
prediction = model.predict()

# Test de diagnostic
residual = model.resid

# Breusch-pagan
lm_statistic, lm_p_value, fvalue, f_p_value = het_breuschpagan(residual, x)
print('Test de Breusch-Pagan')
print('LM Statistic :', lm_statistic)
print('LM p-value :', lm_p_value)
print('F-Statistic :', fvalue)
print('F p-value :', f_p_value)

# Test de Durbin-Watson
dw_statistic = durbin_watson(residual)
print('Test de Durbin-Watson')
print('DW Statistic :', dw_statistic)

# Test de normalitév des résidus
# Test de Shapiro-Wilk
shapiro_statistic, shapiro_p_value = shapiro(residual) 
print('Test de Shapiro-Wilk')
print('Statistic :', shapiro_statistic)
print("P-value :",shapiro_p_value)

# Créer une figure et des sous-graphiques
fig, axs = plt.subplots(4, 1, figsize=(14,20 ))

# Nuage de points pour UBen vs. SocSpend
axs[0].scatter('UBen', 'SocSpend', color='blue', alpha=0.5)
axs[0].set_title('UBen vs. SocSpend')
axs[0].set_xlabel('SocSpend')
axs[0].set_ylabel('UBen')

# Nuage de points pour UBen vs. LFPR
axs[1].scatter('UBen', 'LFPR', color='red', alpha=0.5)
axs[1].set_title('UBen vs. LFPR')
axs[1].set_xlabel('LFPR')
axs[1].set_ylabel('UBen')

# Nauge de point pour UBen vs. CAB_EPL
axs[2].scatter('UBen', 'CAB_EPL', color='green', alpha =0.5 )
axs[2].set_title('UBen vs. CAB_EPL')
axs[2].set_xlabel('CAB_EPL')
axs[2].set_ylabel('UBen')

# Nauge de points pour CAB vs. UBen
axs[3].scatter('UBen', 'CAB', color='black', alpha =0.5 )
axs[3].set_title('UBen vs. CAB')
axs[3].set_xlabel('CAB')
axs[3].set_ylabel('UBen')

# Afficher le graphique
plt.tight_layout()
plt.show()