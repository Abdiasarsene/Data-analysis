# Impotation des librairies importantes
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

# Importation de la base de données 
datatest = pd.read_stata(r'Bases de données/neerspss-français.dta')

# Préliminaires avant l'analyse économétrique
print(
    datatest.head(),
    datatest.tail(),
    datatest.columns,
    datatest.info(),
    datatest.isnull().sum(),
    datatest.duplicated().sum(),
    datatest.select_dtypes(exclude=['object']).drop(columns=('index')).describe()
)

# Visualisation des données
# Matrice de corrélation
correlation = datatest.select_dtypes(exclude=['object']).drop(columns=('index')).corr()
sns.heatmap(correlation,annot=True, cmap='coolwarm')
plt.show()

# Relation entre les variables d"études
fig, axs =plt.subplots(1,2, figsize=(14,5))
sns.scatterplot(x='__ge',y='_SDQ_TotaleProblematics', data = datatest, color = 'purple', alpha=0.5, ax=axs[0])
axs[0].set(
    title ='Relation entre age et TotalProblématic',
    xlabel='age',
    ylabel ='TotalProblematic'
)
sns.scatterplot(x='__ge',y='_SDQ_Subscale_Comportement_proso', data=datatest, color = 'red', alpha=0.5, ax=axs[1]) 
axs[1].set(
    title='Age vs Subscale comportement',
    xlabel='age',
    ylabel='Subscale comportement'
)
plt.tight_layout()
plt.show()

# Visualisation de toutes les variables
sns.pairplot(datatest.select_dtypes(exclude=['object']).drop(columns=('index')))
plt.show()

# Modelisation économétrique
x =datatest.select_dtypes(exclude=['object'])[['_SDQ_TotaleProblematics','_Problemes_SDQ_Subscale_Hyperact','_SDQ_Subscale_Comportement_proso','_KSADS_Symptoms_ODD_TSCORE']] #Variables indépendantes de notre étude
y=datatest.select_dtypes(exclude=['object'])['__ge']

x= sm.add_constant(x) # Ajout de la constante
model= sm.OLS(y,x).fit() # Création du modèle

print(model.summary()) # Affichage des résultats

# Analyse des résidus
# Calcul des résidus
residual = model.resid
# Test de Breusch Pagan
lm_statistic, lm_p_value, f_statistic, f_p_value = het_breuschpagan(residual,x)

alpha = 0.05
if lm_p_value > alpha and f_p_value > alpha:
    print("Le test de Breusch-Pagan est valide, la variance des résidus est constante, ce qui prouve qu'il y a une preuve d'homoscédasticité.")
else:
    print("On rejette l'hypothèse nulle. On peut donc conclure qu'il y a la présence de l'hétéroscédasticité.")

# Test de Shapiro Wilk 
shapiro_test_statistic, p_value_shapiro =shapiro(residual)
if p_value_shapiro > alpha :
    print(
        "On ne rejette pas l'hypothèse nulle. Ainsi, les résidus sont normalement distribués"
    )
else:
    print('L\'hypothèse nulle H0 est rejetée.')
