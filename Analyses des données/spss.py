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

# valeurs manquantes
msno.bar(datasoi, color='blue')

# doublons
doublons=datasoi.duplicated().sum()
print("les doublons sont : ",doublons)

# détection des outliers avec la méthode iqr
Q1 = datasoi.quantile(0.25)
Q3 = datasoi.quantile(0.75)
IQR = Q3-Q1
outliers=((datasoi<(Q1-1.5*IQR))|(datasoi >(Q3+1.5*IQR)))
valeurs_aberrantes=datasoi[outliers.any(axis=1)]

# affichage des résultats
print("les valeurs aberrantes sont : ")
valeurs_aberrantes

# suppression des lignes contenant des outliers
print('Nouvelle base de données sans les outliers :')
datasoi=datasoi.drop(valeurs_aberrantes.index)
datasoi

