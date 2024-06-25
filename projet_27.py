# importation des bibliothèques
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sms
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# importation de la base de données
baseDonnee = pd.read_stata(r'Bases de données/BDD dpae-annuelles-france.dta')

# préparation de la base de données
nbredobservation = baseDonnee.count()#vérification du nombre d'observation
premiereligne = baseDonnee.head() 
derniereligne = baseDonnee.tail()
msno.bar(baseDonnee, color ='purple') # vérification des données manquantes
baseduplicated = baseDonnee.duplicated().sum()# suppression des doublons

# analyse exploration des données (EDA)
#  statistique descriptive
statistic= baseDonnee.drop(columns=['Année']).describe()

# visualisation des données

correlation = baseDonnee.corr()
sns.heatmap(correlation, annot= True, cmap='coolwarm') # visualisation de la corrélation des variables
plt.show() # affichage du graphique

# relation entre la cible et les prédicteurs
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.regplot(data=baseDonnee, x='DPAEbrut', y= 'DonnéeAnnuelles7', color='black', ax=ax[0])
ax[0].set(
    title='Relation entre les données annuelles et le DPAE brut',
    xlabel= 'DPAEbrut',
    ylabel='DonnéeAnnuelles7'
)

sns.regplot(data=baseDonnee, x='DPAEbrut', y='DonnéeAnnuelles7', color='orange', ax=ax[1])
ax[1].set(
    title='Relation entre les données annuelles et le DPAE cvs',
    xlabel='DPAEcvs',
    ylabel='DonnéeAnnuelles7'
)

plt.tight_layout()
plt.show()

# Divison des données ensemble d'entrainemenr et de test
y= baseDonnee['DonnéeAnnuelles7']#variable cible
x= baseDonnee[['DPAEbrut','DPAEcvs']]#variable prédicteur
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.8,random_state=808)

# ajout des constants
x_train_const = sms.add_constant(x_train)
x_test_const = sms.add_constant(x_test)

# entrainement du model
model = sms.OLS(y_train, x_train_const).fit()
print(model.summary())

# prediction
y_pred = model.predict(x_test_const)

# évaluation des performances prédictives
rmse = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# affichage des résultats des métriques de performances
print("Mean Absolute Percentage Error (MAPE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# analyse des résidus
residuals = y_test - y_pred # les résidus pour l'évaluation du modèle

# test d'homoscédasticité
lm_statistic, lm_p_value, f_statistic, f_p_value = het_breuschpagan(residuals, x_test_const)

# affichage des résultat
print("Breusch-Pagan test statistic:", lm_statistic)
print("Breusch-Pagan test p-value:", lm_p_value)
print('fstatisitic :',f_statistic)
print('f_pvalue :',f_p_value)

# test de normaité de résidus
stat, p = shapiro(residuals)
print('stat :',stat)
print('p :',p)

# test d'autocorrélation des résidus
statistic =durbin_watson(residuals)
print('statistic durbin watson : ', statistic)