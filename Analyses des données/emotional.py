# importation des librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRgression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

# imortation du jeu de données au format csv
cherckup=pd.read_stata(r'Bases de données/neerspss-français.dta')


# Exemple de création d'un DataFrame
# Remplacez cette ligne par le chargement de votre DataFrame réel
cherckup = pd.DataFrame(columns=['index', 'participant', '__ge', '_sexe', '_SDQ_TotaleProblematics',
                           '_SDQ_Subscale_EmotionalProblems', '_SDQ_Subscale_ConductProblems',
                           '_Problemes_SDQ_Subscale_Hyperact', '_SDQ_Subscale_ProblemsPeers',
                           '_SDQ_Subscale_Comportement_proso', '_KSADS_Symptoms_ODD_TSCORE',
                           '_KSADS_Diagnosis_ADHD', '_CPRS_RL_A'])

# Dictionnaire pour le renommage des colonnes
rename_dict = {
    'index': 'Index',
    'participant': 'participant',
    '__ge': 'age',
    '_sexe': 'sexe',
    '_SDQ_TotaleProblematics': 'total_SDQ',
    '_SDQ_Subscale_EmotionalProblems': 'emotional_SDQ',
    '_SDQ_Subscale_ConductProblems': 'cond_SDQ',
    '_Problemes_SDQ_Subscale_Hyperact': 'hyper_SDQ',
    '_SDQ_Subscale_ProblemsPeers': 'pairs_SDQ',
    '_SDQ_Subscale_Comportement_proso': 'prosocial_SDQ',
    '_KSADS_Symptoms_ODD_TSCORE': 'ODD_KSADS',
    '_KSADS_Diagnosis_ADHD': 'ADHD_KSADS',
    '_CPRS_RL_A': 'CPRS_A'
}

# Renommer les colonnes du DataFrame
cherckup.rename(columns=rename_dict, inplace=True)

# Afficher les nouvelles colonnes pour vérification

print(cherckup.columns)

"analyse exploratoire des données"

# redéfinition de la base de données
emo= cherckup.select_dtypes(exclude=["object"]).drop(columns=['index'])

emo #affichage du résultats

# affichage des données manquantes
import missingno as msno
msno.bar(cherckup)

# détection des valeurs aberrantes graphiques 
sns.boxplot(data=emo, orient="h", palette="Set2")

# avec la méthode IQR
Q1 = emo.quantile(0.25)
Q3 = emo.quantile(0.75)


IQR = Q3-Q1
outliers=((emo<(Q1-1.5*IQR))|(emo >(Q3+1.5*IQR)))
valeurs_aberrantes=emo[outliers.any(axis=1)]

valeurs_aberrantes #affichage des résultats

# supression des valeurs abérrantes dans le jeu de donnes

emo = emo.drop(valeurs_aberrantes.index)

#statistique descriptive
statistique_emo = emo.describe()
statistique_emo

# matrice de corrélation
correlation_emo = emo.corr()
sns.heatmap(correlation_emo, annot=True, cmap="coolwarm")

#visualisation de la base de données
sns.pairplot(emo, palette="husl")
plt.show()

# Variables prédictrices
variables = ['__ge', '_SDQ_Subscale_EmotionalProblems', '_SDQ_Subscale_ConductProblems', '_Problemes_SDQ_Subscale_Hyperact', '_SDQ_Subscale_ProblemsPeers']

# Création des sous-graphes avec une taille de figure plus grande
fig, axes = plt.subplots(2, 3, figsize=(20, 15))

# Couleurs pour chaque variable
colors = ['red', 'purple', 'green', 'black', 'orange']

# Tracé des graphiques
for ax, var, color in zip(axes.flat, variables, colors):
    sns.regplot(x='_SDQ_TotaleProblematics', y=var, data=emo, scatter_kws={'s': 50}, line_kws={'color': color}, ax=ax)
    ax.set_title(f'Relation entre {var} et _SDQ_TotaleProblematics')
    sns.scatterplot(x='_SDQ_TotaleProblematics', y=var, data=emo, color=color, ax=ax)

# Supprimer le sous-graphe vide
fig.delaxes(axes[1, 2])

# Ajustement de l'espacement entre les sous-graphes
plt.tight_layout()
plt.show()

# NORMALISATION DES DONNEES
min_max_scaler= MinMaxScaler()
emo_normalized= emo.copy()
emo_normalized[variables]= min_max_scaler.fit_transform(emo[variables])
emo_normalized