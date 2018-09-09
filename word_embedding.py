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

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec


''' Função Delete Empty Attributes:
''   Informações:   
''     Autores: Drayton80, ArmandoTGT, douglasliralima
''     Data de Criação: 16/04/2018
''
''   Definição: 
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


''' Função Sentence to Wordlist:
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
''     save_model: se o modelo será salvo em um arquivo ou não (o arquivo salvo é em .txt);
''     save_model_file_name: o nome do arquivo que será salvo o modelo;
''
''   Retorno:
''     Retorna o modelo relativo ao vocabulário de vetores no formato de objeto do Gensim (biblioteca onde se encontram
''      os métodos relativos aos tipos de embedding);
'''
def embedding(data_frame, data_frame_element, embedding_type="word2vec", save_model=False, save_model_file_name="model_vectors"):
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
  # Define o número de dimensões comparativas relativas ao treinamento do modelo
  # Quanto maior esse número, melhor será o treinamento do word2vec
  num_features = 300
  min_word_count = 3
  # Aqui é definido o número de threads que serão rodados baseado na própria CPU,
  # os threads serão invocados para melhorar o desempenho do treinamento
  num_workers = multiprocessing.cpu_count()
  context_size = 7
  downsampling = 1e-3
  seed = 1
  
  if embedding_type == "word2vec" or embedding_type == "Word2Vec" or embedding_type == "w2v":
    model_vectors = Word2Vec(sg=1, seed=seed, workers=num_workers, size=num_features,
                             min_count=min_word_count, window=context_size, sample=downsampling)
    model_vectors.build_vocab(word_sentences)
    model_vectors.train(word_sentences, total_words=len(model_vectors.wv.vocab), epochs=10)
    
  elif embedding_type == "fasttext" or embedding_type == "FastText" or embedding_type == "ft":
    model_vectors = FastText(sg=1, seed=seed, workers=num_workers, size=num_features,
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