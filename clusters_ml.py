# Importation des librairies 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

# Initialisation du clusters
kmean = KMeans(n_clusters=4)
kmean.fit(datastand)
data['Clusters'] = kmean.labels_
data
sns.pairplot(data)
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1],kmean.cluster_centers_[:, 2],kmean.cluster_centers_[:, 3],kmean.cluster_centers_[:, 4], s=300, c='red', label='Centroids')
plt.title('Clusters et Centroids')
plt.legend()
plt.show()

# INTERPRETATION DES RESULTATS
centroids = kmean.cluster_centers_
print('Centroids des clusters:')
print(centroids)