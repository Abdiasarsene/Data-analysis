# Importation de toutes les bases de données
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Récupération de la page web
lien = 'https://astucesfemmes.com/'
resp = requests.get(lien)
html_text = resp.content

# Analyser les données avec du BeautifulSoup
soup = BeautifulSoup(html_text, "html.parser")
text_page = soup.get_text()
print(text_page)