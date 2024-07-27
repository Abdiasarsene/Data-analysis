# Importation des librairies
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import  durbin_watson
from scipy.stats import shapiro

# importation de la base de données
reflet = pd.read_excel(r'c:\Users\ARMIDE Informatique\Downloads\final_data.xlsx')

# Affichage des données du dataset
reflet

"Analyse exploratoire des données"
# affichage des valeurs manquantes
print("Affichages des valeurs manquantes")
msno.matrix(reflet, color='green')

# affichage des doublons
print("Affichages des doublons")
doublons =reflet[reflet.duplicated()]
if not doublons.empty:
    print("Il y a des doublons dans le dataset. Passons illico presto à leur suppression ")
    reflet=reflet.drop_duplicates()
else :
    print("Notre jeu de données ne présente aucuns doublons")

# détection des outliers avec la méthode iqr
Q1 = reflet.quantile(0.25)
Q3 = reflet.quantile(0.75)
IQR = Q3-Q1
outliers=((reflet<(Q1-1.5*IQR))|(reflet >(Q3+1.5*IQR)))
valeurs_aberrantes=reflet[outliers.any(axis=1)]

# affichage des résultats
print("Les valeurs aberrantes sont : ")
valeurs_aberrantes

# statistique descriptive
print("Statistique descriptive du dataset")
reflet.describe()

# visualisation des données

# matrice de corrélation
reflet_corr = reflet.corr()
sns.heatmap( reflet_corr, annot=True, cmap='coolwarm')

# visualisation des variables d'étude
sns.pairplot(reflet, color= 'purple')

# relation entre la variable cible et les prédicteurs
import matplotlib.pyplot as plt
import seaborn as sns

# Création des sous-graphiques
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Premier graphique : Maternal Reflective Functioning vs Prenatal Stress
sns.regplot(data=reflet, x='Prenatal Stress', y='Composite Variable', color='purple', ax=ax[0])
ax[0].set_title('Maternal Reflective Functioning vs Prenatal Stress')
ax[0].set_xlabel('Prenatal Stress')
ax[0].set_ylabel('Maternal Reflective Functioning')

# Deuxième graphique : Maternal Reflective Functioning vs Social Support
sns.regplot(data=reflet, x='Social Support', y='Composite Variable', color='red', ax=ax[1])
ax[1].set_title('Maternal Reflective Functioning vs Social Support')
ax[1].set_xlabel('Social Support')
ax[1].set_ylabel('Maternal Reflective Functioning')

# Ajustement de la mise en page et affichage des graphiques
plt.tight_layout()
plt.show()

# Tracer la régression linéaire
plt.figure(figsize=(10, 6))
sns.regplot(data=reflet, x='Maternal Reflective Functioning', y='Composite Variable', color='red')
plt.title('Maternal Reflective Functioning vs Composite Variable')
plt.xlabel('Maternal Reflective Functioning')
plt.ylabel('Composite Variable')
plt.show()


# création de variable composite avec nos deux variavles cibles : 
# Supposons que 'reflet' est votre DataFrame et les deux variables sont 'Mother-Child Relationship Quality' et 'Child Behavior Problems'

# Création de la variable composite en prenant la moyenne des deux variables
reflet['Composite Variable'] = (reflet['Mother-Child Relationship Quality'] + reflet['Child Behavior Problems']) / 2

# Affichage des premières lignes pour vérifier
reflet[['Mother-Child Relationship Quality', 'Child Behavior Problems', 'Composite Variable']]