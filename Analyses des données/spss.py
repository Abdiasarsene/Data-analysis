# importation des librairies
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_wtason
from sklearn.preprocessing import MinMaxScaler

spss=pd.read_excel(r'c:\Users\ARMIDE Informatique\Desktop\20220609 - (GROUP)-1.xlsx')
print(spss.columns)


# Calculer le score total pour les symptômes d'anxiété
spss['scoreTotalAnxiety'] = spss[["RC_c001.6", "RC_c005.6", "RC_c009.6", "RC_c011.6", "RC_c013.6", "RC_c014.6"
]].sum(axis=1)

# Filtrage des colonnes démarrant par la lettre "S" et suppression des colonnes commençant par le chiffre 7

# Filtrer les colonnes qui commencent par la lettre 'S'
s_columns = [col for col in spss.columns if col.startswith('S')]

# Supprimer les colonnes qui se terminent par le chiffre 7
filtered_columns = [col for col in s_columns if not col.endswith('.7')]

# Afficher les noms des colonnes filtrées
print(filtered_columns)

# Calculer le score total pour l'image de soi
spss['scoreTotalImageSoi'] = spss[[['SC_c004.6', 'SC_c008.6','SC_c012.6','SC_c016.6','SC_c020.6']]].sum(axis=1)

# Calculer le score total pour les conflits dans la relation enseignant-élève
spss['scoreTotalConflit'] = spss[['LL_t001.6', 'LL_t002.6', 'LL_t003.6', 'LL_t004.6', 'LL_t005.6']].sum(axis=1)

'Après le feature engineering, nous créererons un nouveau dataframe.Dataframe qui sera notre nouvelle base de données pour la suite de notre analyse.'

datasoi =spss[['MALE', 'AGE_p.6','scoreTotalConflit','scoreTotalImageSoi','scoreTotalAnxiety']]

# vérification des doublons, des valeurs abérrantes et manquantes

# doublons
doublons=datasoi[datasoi.duplicated()]
print("les doublons sont : ")
doublons

# détection des outliers avec la méthode iqr
Q1 = datasoi.quantile(0.25)
Q3 = datasoi.quantile(0.75)
IQR = Q3-Q1
outliers=((datasoi<(Q1-1.5*IQR))|(datasoi >(Q3+1.5*IQR)))
valeurs_aberrantes=datasoi[outliers.any(axis=1)]

# affichage des résultats
print("les valeurs aberrantes sont : ")
valeurs_aberrantes

# suppression des doublons
datasoi= datasoi.duplicates()

# suppression des lignes contenant des outliers
print('Nouvelle base de données sans les outliers :')
datasoi=datasoi.drop(valeurs_aberrantes.index)
datasoi

# valeurs manquantes
msno.bar(datasoi, color='blue')

# normalisation des données
minmaxscaler =MinMaxScaler()
datasoinorma=datasoi.copy()
datasoinorma[datasoi.columns]=minmaxscaler.fit_transform(datasoi[datasoi.columns])
datasoinorma

# statistique descriptive
statistic= datasoinorma.describe()

# matrice de corrélations
correation =datasoinorma.corr()
sns.heatmap(correation, annot=True, cmap='coolwarm')
plt.show()

# Rélation entre la variabe cible et les préditeurs
variables = ['MALE', 'AGE_p.6', 'scoreTotalConflit','scoreTotalAnxiety']
fig, ax = plt.subplots(2,2, figsize=(18, 9))
# Couleurs pour chaque variable
colors = ['red', 'purple', 'green', 'black', 'orange']

# Tracé des graphiques
for ax, var, color in zip(ax.flat, variables, colors):
    sns.regplot(x='scoreTotalImageSoi', y=var, data=datasoinorma, scatter_kws={'s': 50}, line_kws={'color': color}, ax=ax)
    ax.set_title(f'Relation entre {var} et _SDQ_TotaleProblematics')
    sns.scatterplot(x='_SDQ_TotaleProblematics', y=var, data=datasoinorma, color=color, ax=ax)

# analyse de corrélation  de pearson
x= datasoinorma[['MALE', 'AGE_p.6', 'scoreTotalConflit','scoreTotalAnxiety']] # les prédicteurs

y=datasoinorma.scoreTotalImageSoi

corr, p_value=pearsonr(x,y)
print(f"Corrélation de Pearson :, coor")
print(f"P_value ;, p_value")
