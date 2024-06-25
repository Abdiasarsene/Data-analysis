# Importation des librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Génération des données aléatoires
n = 230
data = {
    'poids en kg': np.random.uniform(12.4, 20, n),  # Correction des bornes
    'age en mois': np.random.randint(130, 500, n),
    'taille': np.random.uniform(1.5, 1.80, n),     # Utilisation de np.random.uniform pour des valeurs flottantes
    'sexe': np.random.choice(["masculin", "feminin"], n)
}

regress_ML = pd.DataFrame(data)
regress_ML.to_excel("ML_GLM.xlsx", index=False)  # Ajout de index=False pour ne pas inclure l'index dans le fichier Excel

# Importation de la base de données
ml= pd.read_csv(r'c:\Users\ARMIDE Informatique\Downloads\advertising.csv')

# Visualisation de la base de données
# Relation entre tv et ventes
fig, ax = plt.subplots(1,2, figsize=(14,6))
sns.regplot(x='tv', y='ventes', data=ml, color='blue', ax=ax[0])
ax[0].set(
    title='Relation entre tv et ventes',
    xlabel='tv',
    ylabel='ventes'
)
# Relation entre radio et ventes
sns.regplot(x='radio', y='ventes', data=ml, color='red', ax=ax[1])
ax[1].set(
    title='Relation entre radio et ventes',
    xlabel='tv',
    ylabel='ventes'
)
plt.tight_layout()
plt.show()

# Relation entre les  journaux et les ventes
sns.regplot(x='journaux', y='ventes', data=ml, color='purple')
plt.show()

# Préparation des données
x=ml[['tv','radio','journaux']] #Prédicteurs
y=ml['ventes'] #Variable cible

# Division des données en ensemble d'entrainement et de test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# Entrainement du model
reg.fit(x_train,y_train)

# Prédiction
y_pred=reg.predict(x_test)

# Evaluation de la performance prédictive
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# Création de nouvelle variable 
ml.tv=ml.tv**2

# Instanciation
scaler = MinMaxScaler()
data_array = scaler.fit_transform(ml)
ml = pd.DataFrame(data_array,
      columns = ['tv','radio','journaux','ventes','tv2'])

# Statisitique descriptive
ml.describe().loc[['min','max']]

# Modelisation et évaluation de la performance prédictive
X = ml[['tv','radio','journaux', 'tv2']]
y = ml.ventes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg.fit(X_train, y_train)
y_hat_test = reg.predict(X_test)

print(f"Coefficients: {reg.coef_}")
print(f"RMSE: {mean_squared_error(y_test, y_hat_test)}")
print(f"MAPE: {mean_absolute_error(y_test, y_hat_test)}")