# Importations des librairies
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Exemple de textes
paragraphes = 'Avec les générations, les études sont devenues de plus en plus longues. Aujourd\'hui un peu moins d`\'un quart de la population de 25 ans ou plus dispose d\'un diplôme de niveau supérieur à au moins bac +2, selon les données de l\'INSEE en 2022. Auparavant les fins d\'études précoces étaient beaucoup plus courantes. Les nouvelles exigences sont-elles ensuite porteuses sur le marché de l\'emploi La Direction de l\'animation de la recherche des études et des statistiques du ministère du travail a publié une étude le 28 mars dernier pour faire le point sur l\'évolution de l\'âge de la sortie d\'études en corrélation avec ses effets sur l\'insertion dans le monde du travail. Elle pose d\'abord le constat que les personnes nées en 1935, soit avant-guerre sont sorties de manière précoce du système scolaire, elles travaillaient souvent avant 17 ans. 40 ans plus tard, en 1975, l`\'âge moyen de fin d\'études est monté à 21 ans. Le niveau est à peu près semblable depuis. Cela s\'explique notamment par la prolongation de l\'obligation d\'aller à l\'école jusqu\'à 16 ans en 1959 et par le développement de formations professionnelles et plus courtes dès la troisième. Les générations nées à partir de la fin des années 1970 vont majoritairement jusqu\'au bac ou diplôme équivalent au minimum (70%). Lequel de ces deux modèles apportent alors le plus d\'avantages au niveau de l\'insertion professionnelle ? Selon l\'étude, "une sortie précoce du système scolaire pourrait apporter un avantage à court terme sur l\'insertion à un âge donné, par rapport aux sortants plus tardifs, du fait d\'une ancienneté plus longue sur le marché du travail, mais ce n\'est pas le cas. À tous les âges, les personnes sorties tôt du système scolaire'

# Téléchargement des biblios
nltk.download('punkt')
nltk.download('wordnet')

# Tokenisation
tokens = word_tokenize(paragraphes)

# Lemmatisation
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

print('')
print('Tokenisation et lemmisation des mots')
print(tokens)
print('')
print(lemmatized_tokens)
print('')