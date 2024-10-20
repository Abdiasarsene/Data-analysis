# Importation des librairies 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Importation de la base de données
data = pd.read_excel(r"d:\Projets\Projet Informatique\Bases de données\secteurs_modified.xlsx")
data = data.select_dtypes(exclude=['object'])
data.columns

data.isna().sum()
data.duplicated().sum()
data= data.drop(columns=['Émissions de CO2 décalées'])
data['Émissions de CO2 décalées'] = data['Log Émissions de CO2']/ data['CO2eq']

# Standardisation des données
scaler = StandardScaler()
datastand = scaler.fit_transform(data)

# Initialisation de l'ACP
acp = PCA(n_components=8)
data_acp = acp.fit_transform(datastand)
pca_data = pd.DataFrame(data=data_acp, columns=["cp1", "cp2", "cp3", "cp4", "cp5","cp6","cp7","cp8"])
sns.pairplot(pca_data)
plt.suptitle('Visualisation de l\'analyse en composante principale')

# INTERPRETER DES RESULTATS
explained_variance = acp.explained_variance_ratio_
print(f'Variance expliquée par la première composante principale: {explained_variance[0]:.2f}')
print(f'Variance expliquée par la deuxième composante principale: {explained_variance[1]:.2f}')
print(f'Variance expliquée par la troisième composante principale: {explained_variance[2]:.2f}')
print(f'Variance expliquée par la quatrième composante principale: {explained_variance[3]:.2f}')
print(f'Variance expliquée par la cinquième composante principale: {explained_variance[4]:.2f}')
print(f'Variance expliquée par la sixième composante principale: {explained_variance[5]:.2f}')
print(f'Variance expliquée par la septième composante principale: {explained_variance[6]:.2f}')
print(f'Variance expliquée par la huitième composante principale: {explained_variance[7]:.2f}')