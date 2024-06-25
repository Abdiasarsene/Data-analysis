import requests
from bs4 import BeautifulSoup

# Importons l'url
url = 'https://perspective.usherbrooke.ca/bilan/servlet/BMTendanceStatPays?langue=fr&codePays=BEN&codeStat=NE.CON.PETC.CD&codeTheme=2'
resp = requests.get(url)
html_content = resp.content

# Récupération des données
soup = BeautifulSoup(html_content, 'html.parser')

# tet = soup.get_text()
H2 = [titre.text.strip() for titre in soup.find_all('h2')]

# Visualisation des données
print(H2)