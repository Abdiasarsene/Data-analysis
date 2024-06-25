import numpy as np
import pandas as pd

# Définir le nombre d'observations et de prédicteurs
n_obs = 200
n_pred = 6

# Générer des prédicteurs aléatoires
np.random.seed(808)
X = np.random.rand(n_obs, n_pred)

# Générer des coefficients pour une relation polynomiale
coefficients = np.random.rand(n_pred, 3)  # Coefficients pour x, x^2, et x^3

# Générer la variable cible avec une relation polynomiale
y = np.zeros(n_obs)
for i in range(n_pred):
    y += coefficients[i, 0] * X[:, i] + coefficients[i, 1] * X[:, i]**2 + coefficients[i, 2] * X[:, i]**3

# Ajouter du bruit à la variable cible
y += np.random.randn(n_obs) * 0.1

# Créer un DataFrame
columns = [f'Pred{i+1}' for i in range(n_pred)]
data = pd.DataFrame(X, columns=columns)
data['Target'] = y

# Afficher un aperçu des données
print(data.head())
