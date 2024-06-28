# # Déclaration des fonctions
# def dataused (x , y) :
#     return x + y
# result = dataused(3 , 5)
# print("Le résulta de la fonction esr : ", result)

# def probability (x,y,z=100) :
#     if y >= 20:
#         return x*y + (x/y*z)
#     else :
#         return x+z+y
# example = probability (12, 5, 6)
# print("Le résultat de la fonction est", example)

# # Retourner plusieurs valeurs
# def func():
#     a=3
#     b=4
#     c=10
#     return a, b, c

# a,b,c = func()
# print(func())

# import numpy as np
# # Scalaire
# used = np.array(1_000_000)
# print(used)

# #Tableau à une dimension
# vector = np.array([2,3,9])
# print(vector)

# # Tableau à deux dimensions
# matrice = np.array([[12,23,111],[3,4,5],[2,4,8]])
# link = np.array([[12,23,1],[23,676,23], [2,4,6]])
# print(matrice)
# print(link)

# some = matrice * link

# #Connaitre les dimensions avec shape
# print (used.shape)
# print (vector.dtype)
# print (matrice.shape)
# print(link[0])

# #Opération avec les ndarray
# print(matrice)
# print(' +')
# print(matrice)
# print(' =')
# print(some)

# import pandas as pd

# # Créer un DataFrame exemple
# data = {'A': [1, 2, 3, 4, 5],
#         'B': [2, 4, 6, 8, 10],
#         'C': [5, 4, 3, 2, 1]}
# df = pd.DataFrame(data)

# # Calculer la covariance
# covariance_matrix = df.cov()
# print(covariance_matrix)

# def abdias(x, y, z):
#         if x <= 0 or y >= 0:
#                 return x/y
#         else :
#                 return x/y + x*z
# print(abdias(-9,-2,3))

# #Importation des données XML
# from lxml import objectify
# import pandas as pd
# path = "datasets/mta_perf/Performance_MNR.xml"
# with open(path) as f:
#         parsed = objectify.parse(f)
# root = parsed.getroot()
# data = []
# skip_fields = ["PARENT_SEQ", "INDICATOR_SEQ",
# "DESIRED_CHANGE", "DECIMAL_PLACES"]
# for elt in root.INDICATOR:
#         el_data = {}
#         for child in elt.getchildren():
#                 if child.tag in skip_fields:
#                         continue
#                 el_data[child.tag] = child.pyval
# data.append(el_data)
# perf = pd.DataFrame(data)
# print(perf.head())