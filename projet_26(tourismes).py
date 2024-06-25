# Importation des librairies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import missingno as msno
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Importation de la bases de données
tours=pd.read_stata(r'Bases de données/BDD_tourisme.dta')

# EDA
# Préparation de la base de données
tours.info()
tours.head()
tours.tail()
tours.isnull().sum()
tours.duplicated().sum()

# Statistique descriptive
stat= tours.describe()

# Visualisation des données
# Matrice de corrélation
correlation=tours.drop(columns=['Années','diff_nbretour', 'diff_toursdep', 'diff_nuit']).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# Rélation entre la variable cible et les prédicteurs
fig,ax=plt.subplots(1,2, figsize=(12,5))
sns.regplot(x='Tauxdecroisanceéconomique', y='Nombredetouristeenmilliers', data=tours, color='red', ax=ax[0])
ax[0].set(
    title='Taux de croissanc économique vs Nombre des touristes',
    xlabel='Taux de croissance',
    ylabel='Nombre des touristes'
)
sns.regplot(x='Tauxdecroisanceéconomique', y='Dépensesdestouristesenmilli', data=tours, color='purple',ax=ax[1])
ax[1].set(
    title='Taux de croissanc économique vs Dépenses des touristes',
    xlabel='Taux de croissance',
    ylabel='Dépenses des touristes'
)

plt.tight_layout()
plt.show()

# Deuxième visualisation
fig,ax=plt.subplots(1,2, figsize=(12,5))
sns.regplot(x='Tauxdecroisanceéconomique', y='Nuitées', data=tours, color='green', ax=ax[0])
ax[0].set(
    title='Taux de croissanc économique vs Nuitées',
    xlabel='Taux de croissance',
    ylabel='Nuitées'
)
sns.regplot(x='Tauxdecroisanceéconomique', y='Recetteglobale', data=tours, color='black',ax=ax[1])
ax[1].set(
    title='Taux de croissanc économique vs Recettes des touristes',
    xlabel='Taux de croissance',
    ylabel='Recettes des touristes'
)
plt.tight_layout()
plt.show()

# Modélisation statistique
# Modèle de régression linéaire 
x =tours[['Nombredetouristeenmilliers', 'Nuitées', 'Recetteglobale','Dépensesdestouristesenmilli']] #Variables prédictrices
y =tours['Tauxdecroisanceéconomique'] #Varibale cible 

# Division des données en ensemble d'entraînement et de test
x_train, y_train, x_test, y_test=train_test_split(x,y,test_size=0.2, random_state=808)

# Entraînement du modèle 
# Ajout des constants
x_train_const= sm.add_constant(x_train)
x_test_const=  sm.add_constant(x_test)
# Création du model
model = sm.OLS(y_train, x_train_const).fit()
print(model.sumùary())

# Analyse des résidus
residuals = model.resid

# Test d'homoscédasticité
lm_statistic, lm_p_value,f_statistic, f_p_value=het_breuschpagan(residuals, x_train_const)

# Affichage des résultats
print('LM Statistic :',lm_statistic)
print('LM pvalue :',lm_p_value)
print('F-statistic :', f_statistic)
print('F pvalue ;',f_p_value)

# Test d'autocorrélation des résidus
statistic_durbin=durbin_watson(residuals)

# Affichage des résultats
print('Durbin-Watson statistic :',statistic_durbin)

# Test de normalité des résidus
statistic_shapiro, p_value_shapiro=shapiro(residuals)

# Affichages des résultats
print('Shapiro-Wilk statistic :',statistic_shapiro)
print('Shapiro-Wilk pvalue :',p_value_shapiro)

# Modélisation prédictive
y_pred =model.predict(x_test)

# Évaluation de la performance du modèle
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print('MAE:', mae)
print('RMSE:', rmse)

# Analyse du modèle
if rmse < 0.1 * y.mean():
    print("Le modèle est bon.")
elif rmse > 0.3 * y.mean():
    print("Le modèle est overfitté.")
else:
    print("Le modèle est underfitté.")