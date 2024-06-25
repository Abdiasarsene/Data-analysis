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
titres_h2 = [titre.text.strip() for titre in soup.find_all('h2')]

# Sélectionner uniquement les 5 premiers titres
titres_h2 = titres_h2[:4]

# # Créer une DataFrame à partir des titres
# data = pd.DataFrame(titres_h2, columns=['Titres'])

# Créer une DataFrame pour stocker les titres et les extraits de 100 mots
data = pd.DataFrame(columns=['Titres', 'Extrait_100_mots'])


# Fonction pour extraire un extrait de 100 mots à partir d'un paragraphe
def extract_100_words(paragraph):
    words = re.findall(r'\b\w+\b', paragraph)  # Séparer le paragraphe en mots
    return ' '.join(words[:100])  # Récupérer les 100 premiers mots et les joindre en une seule chaîne

# Extraire les paragraphes de la page
paragraphes = [paragraphe.text.strip() for paragraphe in soup.find_all('p')]

# Créer une liste pour stocker les données des titres et des extraits de 100 mots
data_rows = []

# Parcourir tous les titres
for i, titre in enumerate(titres_h2):
    # Vérifier si le paragraphe associé existe
    if i < len(paragraphes):
        # Trouver le prochain paragraphe après chaque titre
        paragraphe = paragraphes[i]
        
        # Extraire un extrait de 100 mots à partir du paragraphe
        extrait_100 = extract_100_words(paragraphe)
        
        # Ajouter les titres et les extraits de 100 mots à la liste
        data_rows.append({'Titres': titre, 'Extrait_100_mots': extrait_100})
    else:
        print(f"Aucun paragraphe trouvé pour le titre '{titre}'.")

# Créer une DataFrame à partir de la liste de dictionnaires
data = pd.DataFrame(data_rows)

# Afficher la DataFrame
print(data)