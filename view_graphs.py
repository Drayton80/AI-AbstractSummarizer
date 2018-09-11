import os
import sklearn.manifold
import pandas as pd
import seaborn as sns
import gensim.models.word2vec as w2v
import gensim.models.fasttext as ft
import word_embedding
import matplotlib.pyplot as plt

vectors_word2vec = w2v.Word2Vec.load(os.path.join("vocabulary_models", "vocabulary_word2vec.txt"))
vectors_fasttext = ft.FastText.load(os.path.join("vocabulary_models", "vocabulary_fasttext.txt"))

#word_embedding.vocabulary_graph_2d(vectors_fasttext, "Vetores do Fast Text", graph_label_x="Coordenada X", graph_label_y="Coordenada Y")
#word_embedding.vocabulary_graph_2d(vectors_word2vec, "Vetores do Word2vec", graph_label_x="Coordenada X", graph_label_y="Coordenada Y")

print("Tamanho do Vocabulário Word2Vec:", len(vectors_word2vec.wv.vectors))
print("Tamanho de um Vetor Word2Vec:", len(vectors_word2vec.wv.vectors[0]))
print("Tamanho do Vocabulário Fast Text:", len(vectors_fasttext.wv.vectors))
print("Tamanho de um Vetor Fast Text:", len(vectors_fasttext.wv.vectors[0]))
print("Tamanho dos n-grams:", vectors_fasttext.wv.num_ngram_vectors)




