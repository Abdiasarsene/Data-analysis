import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')

# Texte d'exemple
texte = "Ceci est un exemple de texte. Nous allons calculer la fréquence des mots dans ce texte."

# Tokenisation des mots
mots = word_tokenize(texte)

# Calcul de la fréquence des mots
freq = FreqDist(mots)

# Affichage des mots les plus fréquents
for mot, frequence in freq.items():
    print(f"{mot}: {frequence}")
