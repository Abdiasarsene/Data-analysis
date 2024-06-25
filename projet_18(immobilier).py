# Importation des bibliothèques
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Importation de lazbase de donnée
abdias =pd.read_excel(r"c:\Users\ARMIDE Informatique\Desktop\Formation pratique\bddias.xlsx")

# Creation des données aléatoires pour donner du poids à mon analyse de données
n = 120

data = {
    'indice_prix_logement_lissé': np.random.randint(12, 34, n),
    'volume_transaction': np.random.randint(2, 16, n),
    'taux_occupation': np.random.randint(70, 95, n),
    'taux_prix_logement': np.random.uniform(0.2, 0.3, n),
    'taux_volume_transaction': np.random.uniform(0.2, 0.6, n),
    'indice_prix_logement': np.random.randint(120, 230, n),
    'date': np.random.randint(2001, 2020, n)
}
# Créer ma base de données en mettant les données dans un dataframe
donne_immobilier = pd.DataFrame(data, columns=['indice_prix_logement_lissé','volume_transaction','taux_occupation','taux_prix_logement','taux_volume_transaction','indice_prix_logement','date'])

# Statistique descriptive
stats= donne_immobilier.drop(columns=('date')).describe()

# Visualisation des données
correlation = donne_immobilier.drop(columns=('date')).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()

# Visualisation des relations entre l'indice de logement et les autres variables
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Indice de prix de logement vs Volume de transaction
sns.scatterplot(x='volume_transaction', y='indice_prix_logement', data=donne_immobilier, alpha=0.5, color='red', ax=ax[0])
ax[0].set(
    title='Indice de prix de logement vs Volume de transaction',
    xlabel='Volume de transaction',
    ylabel='Indice de prix de logement'
)
# Indice de prix de logement vs Taux d'occupation
sns.scatterplot(x='taux_occupation', y='indice_prix_logement', data=donne_immobilier, alpha=0.5, color='blue', ax=ax[1])
ax[1].set(
    title='Indice de prix de logement vs Taux d\'occupation',
    xlabel='Taux d\'occupation',
    ylabel='Indice de prix de logement'
)
plt.tight_layout()
plt.show()

# Indice de prix de logement vs Indice de prix de logement lissé
fig, ax = plt.subplots(1,2, figsize=(14,6))
sns.scatterplot(x='indice_prix_logement_lissé',y='indice_prix_logement', data= donne_immobilier, alpha=0.5, color='orange', ax=ax[0])
ax[0].set(
    title='Indice de prix de logement vs Volume de transaction',
    xlabel='Volume de transaction',
    ylabel='Indice de prix de logement lissé'
)
# Indice de prix de logement vs Taux de prix de logement
sns.scatterplot(x='taux_prix_logement',y='indice_prix_logement', data= donne_immobilier, alpha=0.5, color='purple', ax=ax[1])
ax[1].set(
    title='Indice de prix de logement vs Taux de prix de logemment',
    xlabel='Taux de prix de logement',
    ylabel='Indice de prix de logement'
)
plt.tight_layout()
plt.show()

# Préparation des données
X = donne_immobilier[['indice_prix_logement_lissé', 'volume_transaction', 'taux_occupation', 'taux_prix_logement', 'taux_volume_transaction']]
y = donne_immobilier['indice_prix_logement']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajouter une constante pour l'interception
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Ajuster le modèle de régression linéaire
model = sm.OLS(y_train, X_train_const).fit()

# Résumé du modèle
print(model.summary())

# Faire des prédictions
y_pred = model.predict(X_test_const)

# Évaluer le modèle
rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')


