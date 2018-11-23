import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import text_to_word_sequence
import re, os

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec


''' Função Vocabulary Graph 2D:
''   Informações:   
''     Autor: Drayton80
''     Data de Criação: 09/09/2018
''
''   Descrição: 
''     Tal função exibe o gráfico relativo aos vetores de um objeto do gensim que representa um
''     vocabulário de palavras
''
''   Parâmetros de entrada:
''     model_vectors: o objeto do gensim relativo ao vocabulário (ou modelo) de palavras;
''     graph_title: título do gráfico exibido;
''     graph_label_x: texto exibido para descrever o eixo de coordenada horizontal;
''     graph_label_y: texto exibido para descrever o eixo de coordenada vertical;
'''
def vocabulary_graph_2d(model_vectors, graph_title="2D Graph", graph_label_x="X", graph_label_y="Y"):
  # Vai no atributo vectors do objeto aonde contém uma referência
  # para todos os vetores e cada coordenada deles;
  vectors_dimension = len(model_vectors.wv.vectors[0])

  # Se as dimensões dos vetores forem maiores do que 2, é preciso comprimir eles
  # para possibilitar a plotagem 2D
  if 2 < vectors_dimension:
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
  else:
    vectors_matrix_2d = model_vectors.wv.vectors

  vectors_words = []
  vectors_x = []
  vectors_y = []

  # Aqui retira-se as palavras atreladas à cada vetor:
  for word in model_vectors.wv.vocab:
    vectors_words.append(word)

  # Em seguida, é pego as coordenadas de cada um e elas são salvas em listas:
  for vectors in vectors_matrix_2d:
    vectors_x.append(vectors[0])
    vectors_y.append(vectors[1])

  # Usa-se o matplot lib para fazer subplots aonde cada um deles refere-se a um dos vetores em
  # separado (sendo isso feito para eles poderem conter um texto atrelado à eles)
  figure, axes = plt.subplots()
  # Aqui define-se o gráfico como scatter, ou seja, diversos pontos
  axes.scatter(vectors_x, vectors_y)

  # Escreve no gráfico a palavra atrelada ao vetor próxima ao ponto que representa o vetor em si
  for i, word in enumerate(vectors_words):
    axes.annotate(word, (vectors_x[i], vectors_y[i]))

  # Configurações relativas ao próprio matplotlib:
  plt.xlabel(graph_label_x)
  plt.ylabel(graph_label_y)
  plt.title(graph_title)
  plt.show()


''' Função Delete Empty Attributes:
''   Informações:   
''     Autores: Drayton80, ArmandoTGT, douglasliralima
''     Data de Criação: 16/04/2018
''
''   Descrição: 
''     Tal função checa elementos referentes a um atributo em todas as
''     instâncias e deleta aqueles que estiverem vazios, os quais nesse caso
''     são representados como nan (not a number).
''
''   Parâmetros de entrada:
''     data_frame: o data frame o qual será copiado para fazer a eliminação;
''     attribute: o atributo escolhido o qual será feito a checagem 
''      dos elementos;
''
''   Retorno:
''     Retorna o data_frame com as eliminações já feitas.
'''
def delete_empty_attributes(data_frame, attribute):
  new_data_frame = pd.DataFrame(data_frame)
  
  i = 0
  aux = new_data_frame[attribute].tolist()
  
  for element in aux:
    if(pd.isnull(element)):
      print(i,element)
      # Nota: inplace = True significa que a operação feita aqui é tornada
      #       permanente. Por default ele fica False, ou seja, apenas vale
      #       para a linha em que está sendo chamado.
      new_data_frame.drop(i, inplace = True)
      print(new_data_frama[attribute][i])
    i+=1      
  return new_data_frame


''' Função Sentence to Wordlist:
''   Informações:   
''     Autor: Drayton80
''     Data de Criação: 17/08/2018
''
''   Descrição:
''     Transforma uma sentença completa em uma lista de palavras, ou seja, cada
''     palavra se torna um elemento dessa lista retornada.
''
''   Parâmetros de Entrada:
''     sentence: a sentença recebida a qual as palavras serão separadas;
''
''   Retorno:
''     Retorna a lista de palavras relativa a sentença;
'''
def sentence_to_wordlist(sentence):
  clean = re.sub("[^a-zA-Z]"," ", str(sentence).lower())
  words = clean.split()
  
  return words


''' Função Embedding:
''   Informações:   
''     Autor: Drayton80
''     Data de Criação: 08/09/2018
''
''   Descrição:
''     Transforma um data frame em um vocabulário de vetores usando uma determinada técnica específica para tal
''
''   Parâmetros de Entrada:
''     data_frame: o próprio data frame a ser convertido;
''     data_frame_element: como o data frame é um dicionário, é preciso especificar a palavra (ou elemento) ao qual será
''      aplicado a conversão;
''     embedding_type: o string relativo ao tipo de embedding que será aplicado na conversão;
''     vector_dimension: dimensão de cada vetor individual;
''     save_model: se o modelo será salvo em um arquivo ou não (o arquivo salvo é em .txt);
''     save_model_file_name: o nome do arquivo que será salvo o modelo;
''
''   Retorno:
''     Retorna o modelo relativo ao vocabulário de vetores no formato de objeto do Gensim (biblioteca onde se encontram
''      os métodos relativos aos tipos de embedding);
'''
def embedding(data_frame, data_frame_element, embedding_type="word2vec", vector_dimension=50,
              save_model=False, save_model_file_name="model_vectors"):
  print("<STARTING(embededding_words)>")
   
  # TRANSFORMAÇÂO DE PALAVRAS EM TOKENS:
  nltk.download("punkt")
  nltk.download("stopwords")

  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  
  raw_sentences = []
  
  # Se algum dos elementos não possuir qualquer texto, ou seja, for demarcado como NAN
  # (que é o caso desse data set), as linhas serão descardadas do data frame
  for line in range(len(data_frame)-1):
    #print(line, data_frame['ctext'][line])
    # Ignora os elementos com NAN
    if not pd.isnull(data_frame[data_frame_element][line]):
      # Retira-se cada linha referente aos textos completos e sumarizados e aplica o tokenizer
      # neles (PESQUISAR O QUE O TOKENIZER FAZ)
      raw_sentences.append(tokenizer.tokenize(str(data_frame[data_frame_element][line])))
    
  word_sentences = []
  
  for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
      word_sentences.append(sentence_to_wordlist(raw_sentence))
  
  
  # APLICAÇÂO DO MODELO DE EMBEDDING:
  min_word_count = 3
  # Aqui é definido o número de threads que serão rodados baseado na própria CPU,
  # os threads serão invocados para melhorar o desempenho do treinamento
  num_workers = multiprocessing.cpu_count()
  context_size = 7
  downsampling = 1e-3
  seed = 1
  
  if embedding_type == "word2vec" or embedding_type == "Word2Vec" or embedding_type == "w2v":
    model_vectors = Word2Vec(sg=1, seed=seed, workers=num_workers, size=vector_dimension,
                             min_count=min_word_count, window=context_size, sample=downsampling)
    model_vectors.build_vocab(word_sentences)
    model_vectors.train(word_sentences, total_words=len(model_vectors.wv.vocab), epochs=10)
    
  elif embedding_type == "fasttext" or embedding_type == "FastText" or embedding_type == "ft":
    model_vectors = FastText(sg=1, seed=seed, workers=num_workers, size=vector_dimension,
                             min_count=min_word_count, window=context_size, sample=downsampling)
    model_vectors.build_vocab(word_sentences)
    model_vectors.train(word_sentences, total_words=len(model_vectors.wv.vocab), epochs=10)
    
  else:
    # Exibe uma mensagem de erro avisando que o tipo escolhido para o embedding
    # não corresponde aqueles encontrados na função
    print("<ERROR: Embedding type not found in function list >")
    return
  
  if save_model:
    if not os.path.exists("vocabulary_models"):
      os.makedirs("vocabulary_models")
    
    model_vectors.save(os.path.join("vocabulary_models", "{0}.txt".format(save_model_file_name)))
    
    print("<MESSAGE: Embedding Model has been successfully saved >")
    
  print("<COMPLETE(embededding_words)>")
  
  return model_vectors


def sentence_clean(sentence):
  clean = re.sub("[^a-zA-Z]"," ", str(sentence).lower())
  
  return clean


def tokenizer(data_frame, data_frame_element, max_sequence_length=10000):    
  # Faz uma limpeza no texto para deixá-lo corretaente ajustado:
  # Nota: Criar funções para isso 
  #data_frame[data_frame_element] = data_frame[data_frame_element].apply(lambda x: x.lower())
  #data_frame[data_frame_element] = data_frame[data_frame_element].apply(lambda x: clean_str(x))
  #data_frame[data_frame_element] = data_frame[data_frame_element].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

  nltk.download('stopwords')

  # Separa um conjunto de stopwords (inglês) fornecido pelo NLTK:
  stop_words = set(stopwords.words('english'))

  # Cria uma lista vazia aonde ficarão os textos:
  text = []

  # Retira-se as stopwords do texto:
  for row in data_frame[data_frame_element].values:
    words_clean = sentence_clean(row)
    word_list = text_to_word_sequence(words_clean)
    no_stop_words = [w for w in word_list if not w in stop_words]
    no_stop_words = " ".join(no_stop_words)
    text.append(no_stop_words)

  # Aqui é usado o Tokenizer do próprio Keras para transformar as palavras em tokens
  tokenizer = Tokenizer(split=' ')

  tokenizer.fit_on_texts(text)

  text_tokenized = tokenizer.texts_to_sequences(text)  
  
  text_tokenized = pad_sequences(text_tokenized, maxlen=max_sequence_length)

  return text_tokenized

def preprocess_data(data_frame, save_data=False, save_data_file_name="preprocessed_data"):
  # Aplicando o pré-processamento do texto e, em seguida, o Tokenizer do Keras pertencente ao Tensorflow
  # é possível converter cada palavra em tokens e, por fim, os textos em conjunto de tokens
  x = tokenizer(data_frame, 'ctext')
  y = tokenizer(data_frame, 'text')
  # Como o retorno do Tokenizer é um nparray é necessário convertê-lo em uma lista para ser possível posteriormente usá-lo
  # para criar um data frame do pandas
  x = x.tolist()
  y = y.tolist()

  print(len(min(x)))
  print(len(max(x)))
  print(len(min(y)))
  print(len(max(y)))

  # Caso save_data seja verdadeiro, o data frame é salvo:
  if save_data is True:
    # Aqui é criado uma pasta chamada vocabulary_models caso ela já não exista:
    if not os.path.exists("vocabulary_models"):
      os.makedirs("vocabulary_models")

    # Cria-se um data frame do pandas passando como parâmetro um dicionário formado pelas listas x e y
    data_frame_preprocessed = pd.DataFrame({'x': x, 'y':y})
    # Tal data frame é salvo no formato de um arquivo csv com o nome fornecido:
    data_frame_preprocessed.to_csv("vocabulary_models/{0}.csv".format(save_data_file_name))

  return data_frame_preprocessed
