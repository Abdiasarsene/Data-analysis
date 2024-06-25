# Importatin des librairies
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    print('')
    print('LE WEBSCRAPPING : NLP')

    # Premier scrapping et exportation des données 
    print('')
    print('Scrapping sur le deuxième lien')
    print('')

    # Importation de l'url
    url ='https://www.futura-sciences.com/sante/actualites/cerveau-consequences-insoupconnees-alimentation-quartiers-defavorises-107815/'
    respo = requests.get(url)
    html_content = respo.content

    # Extraction des données
    soup = BeautifulSoup(html_content, 'html.parser')

        # Futura-h2
    title_text = [ titre.text.strip() for titre in soup.find_all('h2')]
    print(title_text)
    print('')

        # Futura-p
    p_text = [ paragraphe.text.strip() for paragraphe in soup.find_all('p')]

    # Téléchargement des biblios
    nltk.download('punkt')
    nltk.download('wordnet')

    # Tokenisation
    tokens = word_tokenize(p_text)

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    print('Tokenisation et lemmisation des mots')
    print(lemmatized_tokens)
    print('')

    # Deuxième scrapping et exportation des données
    print('')
    print('Scrapping sur le deuxième lien')
    print('')

    # Importation du lien à scraper
    lien = 'https://astucesfemmes.com/decouverte-mode-pour-shabiller-a-moindre-cout-les-pepites-de-la-marque-zara/'
    resp = requests.get(lien)
    content_html = resp.content

    #  Récupération des données 
    soup = BeautifulSoup(content_html, 'html.parser')

    # Astucesfemme-H2
    title = [titre.text.strip() for titre in soup.find_all('h2')]
    print(title)
    print('')

    # Astucesfemmes-p
    para = [paragraphe.text.strip() for paragraphe in soup.find_all('p')]
    print(para)
    print('')

    # Téléchargement des biblios
    nltk.download('punkt')
    nltk.download('wordnet')

    # Tokenisation
    tokens = word_tokenize(para)

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    print('Tokenisation et lemmisation des mots')
    print(lemmatized_tokens)
    print('')
except Exception as e : 
    print('Erreur : ', e)