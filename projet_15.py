# Importation des librairies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2

# Importation de la base de données
DonneSante = pd.read_excel('DonneSante.xlsx')

# Statistique descriptive
Sante = DonneSante.select_dtypes(exclude=['object'])
descript_Sante = Sante.drop(columns=['Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0']).describe()

# Visualisation des données
# Graphique pour voir la relation entre les variables
# Définir le thème
sns.set_theme(style='whitegrid')

# Créer une figure avec deux sous-graphiques
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Premier scatter plot
sns.scatterplot(x='age', y='Arthrose_code', data=Sante, alpha=0.5, color='green', ax=axs[0])
axs[0].set(title='Age vs Arthrose', xlabel='Age', ylabel='Arthrose')

# Deuxième scatter plot (exemple supplémentaire)
sns.scatterplot(x='age', y='taux_de_cholesterol', data=Sante, alpha=0.5, color='blue', ax=axs[1])
axs[1].set(title='Age vs Taux de Cholestérol', xlabel='Age', ylabel='Taux de Cholestérol')

# Ajuster la mise en page / Afficher la figure avec les deux graphiques
plt.tight_layout()
plt.show()

# matrice de Corrélation
corre = DonneSante.select_dtypes(include=['int64']).corr()
# Visualisation de la matrice de corrélation
sns.heatmap(corre, annot=True, cmap='Blues', fmt='.2f')
plt.show()

# Analyse économétrique
# Supposons que DonneSante est votre DataFrame et 'y' est la variable cible binaire
X = DonneSante[['taux_de_cholesterol', 'age', 'taux_de_glycemie','pression_arterielle','Medicament_code']]  # vos variables explicatives
y = DonneSante['Arthrose_code']  # votre variable cible

# Ajout d'une constante
X = sm.add_constant(X)

# Estimation du modèle logistique
logit_model = sm.Logit(y, X).fit()

# Résumé du modèle
print(logit_model.summary())

# Test de Wald
# Les p-values dans le résumé du modèle indiquent les résultats du test de Wald pour chaque coefficient.

# Test du Rapport de Vraisemblance
llr_pvalue = logit_model.llr_pvalue
print(f'LLR p-value: {llr_pvalue}')

# Test d'Hosmer-Lemeshow
# Pour ce test, vous pouvez utiliser la bibliothèque 'sklearn'

# Prédictions de probabilités
y_pred_prob = logit_model.predict(X)

# Découpage des probabilités en 10 groupes
data = pd.DataFrame({'y': y, 'y_pred_prob': y_pred_prob})
data['group'] = pd.qcut(data['y_pred_prob'], 10, duplicates='drop')
grouped = data.groupby('group')

# Calcul des statistiques observées et attendues pour chaque groupe
observed = grouped['y'].sum()
expected = grouped['y_pred_prob'].sum()
hosmer_lemeshow_stat = ((observed - expected) ** 2 / (expected * (1 - expected / len(data)))).sum()

# Calcul de la p-value du test d'Hosmer-Lemeshow
hosmer_lemeshow_pvalue = 1 - chi2.cdf(hosmer_lemeshow_stat, df=8)
print(f'Hosmer-Lemeshow test p-value: {hosmer_lemeshow_pvalue}')
