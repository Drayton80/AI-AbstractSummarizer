import numpy as np
import pandas as pd
import scipy
import sklearn
import string
from tqdm import tqdm
import os

#data_path = "/home/pedroabrantes/Sumarizacao/codigos/written/"
data_path = "written"

original = []
compressed = []

for path in tqdm(os.listdir(data_path)):
	#print(path)
	if path != '.rlog':
		with open(data_path + "/" + path, 'r', encoding='utf-8') as f:
			lines = f.read().split('\n')

		aux = []
		aux2 = []

		#len(linhas) - 2
		#fazer com range 8 pra testar
		for i in range (2, len(lines) - 2):
			aux = lines[i].split('>', 1)
			aux2 = aux[1].split('<', 1)
			if "<original" in aux[0]:
				original.append(aux2[0])
			if "<compressed" in aux[0]:
				compressed.append(aux2[0])

print("tamanho lista original: ", len(original))
print("tamanho lista compressed: ", len(compressed))

'''	print("\n\nOriginal phrases:")
	for i in range(len(original)):
		print(i, original[i])

	print("\n\nCompressed phrases:")
	for i in range(len(compressed)):
		print(i, compressed[i])

print("tamanho lista original: ", len(original))
print("tamanho lista compressed: ", len(compressd))

#descomentar depois
'''

'''
with open(data_path, 'r', encoding='utf-8') as f:
	lines = f.read().split('\n')

aux = []
aux2 = []

#len(linhas) - 2
#fazer com range 8 pra testar
for i in range (2, len(linhas) - 2):
	aux = lines[i].split('>', 1)
	aux2 = aux[1].split('<', 1)
	if "<original" in aux[0]:
		original.append(aux2[0])
	if "<compressed" in aux[0]:
		compressed.append(aux2[0])

print("\n\nOriginal phrases:")
for i in range(len(original)):
	print(i, original[i])

print("\n\nCompressed phrases:")
for i in range(len(compressed)):
	print(i, compressed[i])
'''