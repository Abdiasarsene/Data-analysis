# Importation des librairies
import pandas as pd
import missingno as msno
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# Importation de la base de données
esgdata = pd.read_csv(r'c:\Users\ARMIDE Informatique\Downloads\goodwill-esg_data.csv')

# affichage de la base de données
esgdata

"Exploratory analysis data"

# vérification des données manquantes de notre base de données
msno.bar(esgdata, color ='purple')

# vérification des doublons dans le jeu de données

dupli= esgdata[esgdata.duplicated()] #affichage des doublons
if not dupli.empty:
    print("There are duplicates in the dataset") 
    dupli.drop_duplicates() # suppression des doublons
else:
    print("There are not duplicates in the dataset")

# détection des outliers
# détection des outliers avec la méthode iqr
Q1 = esgdata.quantile(0.25)
Q3 = esgdata.quantile(0.75)
IQR = Q3-Q1
outliers=((esgdata<(Q1-1.5*IQR))|(esgdata >(Q3+1.5*IQR)))
valeurs_aberrantes=esgdata[outliers.any(axis=1)]

# affichage des résultats
print("The outliers are: ")
valeurs_aberrantes

# statistique descriptive
print("Descriptive statistics of the dataset")
esgdata.describe()

# visualisation des données

# matrice de corrélation
esg_corr = esgdata.corr()
sns.heatmap(esg_corr, annot=True, cmap='coolwarm')

# visualisation des variables d'étude
sns.pairplot(esgdata)

# Création des sous-graphiques
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Premier graphique : Maternal Reflective Functioning vs Prenatal Stress
sns.regplot(data=esgdata, x='ESG_Score', y='Goodwill', color='purple', ax=ax[0])
ax[0].set_title('Goodwill vs ESG_Score')
ax[0].set_xlabel('ESG_Score')
ax[0].set_ylabel('Goodwill')

# Deuxième graphique : Maternal Reflective Functioning vs Social Support
sns.regplot(data=esgdata, x='Log_Asset_Total', y='Goodwill', color='red', ax=ax[1])
ax[1].set_title('Goodwill vs Log_Asset_Total')
ax[1].set_xlabel('Log_Asset_Total')
ax[1].set_ylabel('Goodwill')

# Ajustement de la mise en page et affichage des graphiques
plt.tight_layout()
plt.show()

# Tracer la régression linéaire
plt.figure(figsize=(12, 6))
sns.regplot(data=esgdata, x='Asset_Turnover', y='Goodwill', color='orange')
plt.title('Goodwill vs Asset_Turnover')
plt.xlabel('Asset_Turnover')
plt.ylabel('Goodwill')
plt.show() 

# normalisation des données
minmaxscaler =MinMaxScaler()
esgdatanorma=esgdata.copy()
esgdatanorma[esgdata.columns]=minmaxscaler.fit_transform(esgdata[esgdata.columns])
esgdatanorma

# analyse de corrélation de pearson
corr_pearson=esgdatanorma.corr(method= 'pearson')

# analyse de la médiation
# Définir les variables
X = esgdatanorma[['ESG_Score', 'Log_Asset_Total', 'Asset_Turnover']] # les prédicteurs
Y = esgdatanorma['Goodwill'] #variable cible 

# Ajoutez une constante (intercepte)
X = sm.add_constant(X)

# Modèle de régression
model1 = sm.OLS(Y, X).fit()
print(model1.summary())
