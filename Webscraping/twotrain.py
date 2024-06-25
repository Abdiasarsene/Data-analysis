import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Récupérer le contenu HTML de la page 
url = "https://astucesfemmes.com/comment-reussir-son-premier-rendez-vous-amoureux/"				
response = requests.get(url)
html_content = response.content

# Analyser le contenu HTML avec BeautifulSoup
soup =  BeautifulSoup(html_content, 'html.parser')

# Extraire tous les titres H2 de la page 
# titres_h2 = [titre.text.strip() for titre in soup.find_all('h2')]

paragraphe_text = [ paragraphe.text.strip() for paragraphe in soup.find_all("p")] 

# Afficher les 03 premiers paragraphes

paragraphe_text = paragraphe_text[:3]

# Récupérer ses données dans une  dataframe

# datapara = pd.DataFrame( columns =["Paragraphe"])

print(paragraphe_text)

