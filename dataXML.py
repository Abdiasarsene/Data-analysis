from lxml import etree

# Définition d'un exemple de document XML
xml_data = """
<root>
    <person>
        <name>John Doe</name>
        <age>30</age>
        <city>New York</city>
    </person>
    <person>
        <name>Jane Smith</name>
        <age>25</age>
        <city>Los Angeles</city>
    </person>
</root>
"""

# Analyse du document XML
tree = etree.fromstring(xml_data)

# Récupération des informations
for person in tree.xpath('/root/person'):
    name = person.xpath('name/text()')[0]
    age = person.xpath('age/text()')[0]
    city = person.xpath('city/text()')[0]
    print(f"Nom : {name}")
    print(f"Âge : {age}")
    print(f"Ville : {city}")


#Importation  de la bibliothèque XML
from lxml import etree

#Chargement  du document XML
tree = etree.parse('chemin_vers_votre_fichier.xml')
tree = etree.fromstring(xml_data)

#Navigation dans la structure XML
elements = tree.xpath('/chemin_vers_votre_element')

#Extraction des données
for element in elements:
    text = element.xpath('chemin_vers_votre_texte')[0]
    attribute = element.get('attribut')

#Manipulation et traitement des données
# Stockage des données dans une liste de dictionnaires
data = [{'text': text, 'attribut': attribute} for element in elements]

#Gestion des erreurs
try:
    # Tentative de sélection des éléments XML
    elements = tree.xpath('/chemin_vers_votre_element')
except etree.XPathSyntaxError:
    # Gestion de l'erreur de syntaxe XPath
    print("Erreur de syntaxe XPath.")