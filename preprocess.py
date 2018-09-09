import pandas as pd
import word_embedding

# Atribui a um data frame os dados relativos ao arquivo csv do data set:
#   Nota: é preciso colocar o encoding como latin1 pois o utf8 não aceita certos
#         caracteres que estão presentes no texto da base de dados
data_frame = pd.read_csv("datasets/news_summary.csv", encoding = 'latin1')
# Retira do data frame os atributos (colunas) citados abaixo como parametro:
#   Nota 1: axis = 1 significa que a operação feita aqui é executada em uma 
#           coluna do data frame.
#   Nota 2: inplace = True significa que a operação feita aqui é tornada
#           permanente. Por default ele fica False, ou seja, apenas vale
#           para a linha em que está sendo chamado.
data_frame.drop(['author', 'date', 'read_more', 'headlines', 'text'], axis = 1, inplace = True)

# Usa a função embedding words e salva o resultado do modelo em um arquivo:
word_embedding.embedding(data_frame, "ctext", embedding_type="word2vec", save_model=True, save_model_file_name="vocabulary_word2vec")
word_embedding.embedding(data_frame, "ctext", embedding_type="fasttext", save_model=True, save_model_file_name="vocabulary_fasttext")