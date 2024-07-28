import requests
import pandas as pd

# Define the URL
url = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-analyst/Cours_realisez_des_modelisations_performantes/NOUVEAU/Dataset_ozone.txt"

# Send a GET request to the URL
response = requests.get(url)

# Raise an exception if the request was unsuccessful
response.raise_for_status()

# Convert the response text to a pandas DataFrame
from io import StringIO

data = StringIO(response.text)
df = pd.read_csv(data, delimiter=";")  # Adjust delimiter if necessary












# Display the first few rows of the DataFrame
print(df.head())
import pandas as pd
import numpy as np
import random

# Génération de données fictives
n_samples = 1000  # Nombre total d'échantillons

# Initialisation des listes pour stocker les données
data = {
    'Goodwill': [],
    'ESG_Score': [],
    'Log_Asset_Total': [],
    'Asset_Turnover': []
}

# Génération des données fictives
for _ in range(n_samples):
    # Goodwill: valeurs fictives entre 1 et 10 (en millions)
    data['Goodwill'].append(random.uniform(1, 10))
    
    # ESG Score: scores fictifs entre 0 et 100
    data['ESG_Score'].append(random.uniform(0, 100))
    
    # Log Asset Total: valeurs fictives de logarithmes entre 10 et 20
    data['Log_Asset_Total'].append(random.uniform(10, 20))
    
    # Asset Turnover: ratios fictifs entre 0.1 et 2
    data['Asset_Turnover'].append(random.uniform(0.1, 2.0))

# Conversion en DataFrame pandas
df = pd.DataFrame(data)

# Affichage du DataFrame final
print(df.head())

# Enregistrement du DataFrame dans un fichier CSV
df.to_csv('goodwill-esg_data.csv', index=False)

