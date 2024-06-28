# Importation des librairies d'études
import requests
from bs4 import BeautifulSoup
import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Importons l'url
url = 'https://astucesfemmes.com/comment-reussir-son-premier-rendez-vous-amoureux/'
resp = requests.get(url)
html_content = resp.content

# Récupération des données
soup = BeautifulSoup(html_content, 'html.parser')

try: 
# Récupération des titres et des paragraphes
    paragraphe = [paragraph.text.strip() for paragraph in  soup.find_all('p')]
    paragraphe = paragraphe[:5]
    H2 = [titre.text.strip() for titre in soup.find_all('h2')]
    H2= H2[:5]
except Exception as e : 
# Affichage des erreurs
    print("Erreur", e)

# Création d'une base de données textuelle
# Création du dictionnaire
data = {
    'titre' : H2,
    'paragraphe' : paragraphe
}

# Création du dataframe
text_data = pd.DataFrame(data)
print(text_data)
# try :
#     # Téléchargement des ressources NLTK nécessaires
#     nltk.download('punkt')

#     # Tokenisation des mots pour chaque paragraphe
#     mots_paragraphe = [word_tokenize(paragraphe) for paragraphe in text_data['paragraphe']]

#     # Concaténation de toutes les listes de mots
#     mots = [mot for sublist in mots_paragraphe for mot in sublist]

#     # Calcul de la fréquence des mots
#     freq = FreqDist(mots)

#     # Affichage des mots les plus fréquents
#     for mot, frequence in freq.items():
#         print(f"{mot}: {frequence}")
# except Exception as e :
# # Affichage des erreurs
#     print('Erreur : ', e)
