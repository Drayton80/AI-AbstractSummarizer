# coding: utf-8
# A linha acima foi usada pois estava acontecendo um erro de compilação relativa a falta de caracteres
# relativos ao ASCII no código

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
data_frame.drop(['author', 'date', 'read_more', 'headlines'], axis = 1, inplace = True)

# Usa a função embedding words e salva o resultado do modelo em um arquivo:
#word_embedding.embedding(data_frame, "ctext", embedding_type="word2vec", vector_dimension=300, save_model=True, save_model_file_name="vocabulary_word2vec")
#word_embedding.embedding(data_frame, "ctext", embedding_type="fasttext", vector_dimension=300, save_model=True, save_model_file_name="vocabulary_fasttext")

x = word_embedding.tokenizer(data_frame, 'ctext')
y = word_embedding.tokenizer(data_frame, 'text')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

data_preprocessed = {'x_train': x_train, 'x_test': x_test,
                     'y_train': y_train, 'y_test': x_test}

if not os.path.exists("vocabulary_models"):
  os.makedirs("vocabulary_models")

model_vectors.save(os.path.join("vocabulary_models", "preprocessed_data.txt"))

