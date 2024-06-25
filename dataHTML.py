#Importation des bibliothèques
import requests
import pandas as pd

#Définition de l'url de l"API GitHub
url = "https://api.github.com/repos/pandas-dev/pandas/issues"

#Effectuer une requête HTTP GET
resp = requests.get(url)

#Véritage si la réponse est en cas d'erreur (status code != 2
resp.raise_for_status()
print(resp)

#Conversion du json en Python
data = resp.json()
issues = pd.DataFrame(data, columns=["number","title","labels", "state"])
# print(issues)

f = issues.count()
# print(f)

# #Afficher une colonne spécifique
# htm= pd.DataFrame(data, columns=['labels', 'state'])
# print(htm.head())
