'''
# Para poder plotar o gráfico, é preciso comprimir o número de dimensões o que
# será feito utilizando o método de machine learning chamado TSNE 
# (T Stochastic distributed Neighbour Embedding). Tal método, em linhas gerais, 
# comprime as dimensões (nesse caso, 300) em um número bem menor de dimensões 
# (neste caso definido por n_components = 2).
# Nessa linha abaixo apenas é inicializado o TSNE:
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

# .wv.vectors é um atributo que contém uma lista com todos os vetores do vocabulário,
# ou seja, uma lista de lista de coordenadas. Já fit_transform faz a conversão das n
# dimensões para o novo número especificado antes
vectors_matrix_2d = tsne.fit_transform(model_vectors.wv.vectors)

'''
vectors_matrix_2d = [[2,0],[3,10]]

with open("vocabulary_fasttext_2d.txt", "a") as file:

