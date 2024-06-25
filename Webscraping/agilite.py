# Importation des bibliothèques 
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Importation de l'url
url='https://www.atlasmanagement.nc/les-origines-de-lagilite-partie-1/'
resp = requests.get(url)
datatext = resp.content

# Extraction des contenus sur la page web
soup  = BeautifulSoup( datatext, 'html.parser')

# Importation des H2
titreh2 =[titre.text.strip() for titre in soup.find_all('h2')]
titreh2 = titreh2[: 3]

# Importation de trois paragraphes
para =[paragraph.text.strip() for paragraph in soup.find_all('p')]

paras = pd.DataFrame(para, columns = 'Paragraphes')

print(paras)

# # Vérifier si le nombre de paragraphes est supérieur ou égal au nombre de titres h2
# if len(para) == len(titreh2):
#     # Zipper les titres et les paragraphes
#     data = list(zip(titreh2, para[:len(titreh2)]))
# else:
#     # Si le nombre de paragraphes est inférieur, tronquer la liste des paragraphes
#     data = list(zip(titreh2[:len(para)], para))

# # Créer un DataFrame
# df = pd.DataFrame(data, columns=['Titre', 'Paragraphe'])

# # Visualisation des données
# print(df)
# print(titreh2)
# print(para)