# Importation des bibliothèques
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

#  Récupération de la page HTML du site web
url = 'https://astucesfemmes.com/'
resp = requests.get(url)
html_titre = resp.content

# Analyser le contenu HTML avec du BeautifulSoup
soup = BeautifulSoup(html_titre, 'html.parser')

# Extraire les données textuelles sur la page web
titre_h2 =  [titre.text.strip() for titre in soup.find_all('h2')]

para = [paragraph.text.stripè for paragraph in soup.find_all('p')]

# Extraire les 03 premiers h2
titre_h2 = titre_h2[:3]
para = para[:3]

# Afficher les résultats dans une Dataframe
print("Titres de la page ", titre_h2)
print("Les paragraphes sont: ", para)