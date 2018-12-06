import pandas as pd
import numpy as np
import os
import re

import preprocess
import learning_models
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

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

# Coso os dados ainda não tenham sido pré-processados:
if not os.path.exists('vocabulary_models/preprocessed_data.csv'):
	# Aplica-se o pré-processamento neles e é criado um arquivo. Além de ser obtido o data frame:
	data_frame_preprocessed = preprocess.preprocess_data(data_frame, save_data=True)

	x = data_frame_preprocessed['x'].values.tolist()
	y = data_frame_preprocessed['y'].values.tolist()
else:
	# Caso contrário o arquivo é aberto no formato de data frame:
	data_frame_preprocessed = pd.read_csv('vocabulary_models/preprocessed_data.csv')

	# Transforma os valores no data frame em listas para
	x_string = data_frame_preprocessed['x'].values.tolist()
	y_string = data_frame_preprocessed['y'].values.tolist()

	x = []
	y = []

	# Aqui é feito um pré-processamento para retirar os "[" e "]", pois no momento em que o data frame é
	# convertido para CSV as listas mantém tais caracteres no string do arquivo, ou seja, uma lista
	# [12, 2,..., 0] era convertida para "[12, 2,..., 0]" no arquivo, ao em vez de "12, 2,..., 0", que
	# seria a forma correta:
	for text in x_string:
		text = re.sub("\]|\[", "", text)

		x.append(list(map(int, text.split(", "))))

	for text in y_string:
		text = re.sub("\]|\[", "", text)

		y.append(list(map(int, text.split(", "))))

# Pega o texto com maior tamanho dentre os textos:
max_sequence_length = 130

model = learning_models.keras_lstm(max_sequence_length, 5000, vectors_dimension=300, tensorflow_gpu=True)

model_optimizer = Adam(lr = 0.0001, decay = 0.00000)

model.compile(loss = 'binary_crossentropy', optimizer=model_optimizer, metrics = ['accuracy'])

#print(len(min(x)))
#print(len(max(x)))
#print(len(min(y)))
#print(len(max(y)))

# em seguida, transformá-los em arrays do numpy
x = np.array(x)
y = np.array(y)
# Sendo isso necessário, pois os parâmetros de entrada da função train_test_split
# (pertencentes a biblioteca do sklearn) devem ser arrays do numpy.
# Tal função é usada para dividir a base de dados em teste e treino.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 42)

if not os.path.exists('./model/model_saved.txt'):

    model.fit([x_train, y_train], y_train, validation_data=([x_test, y_test], y_test),
              epochs=100, batch_size=128, shuffle=True, verbose=2)

    model.save_weights('model/model_saved.txt')    

else:
    model.load_weights('model/model_saved.txt')

#scores = model.evaluate(x_test, y_test, verbose = 0, batch_size = 32)
#print("Acc: %.2f%%" % (scores[1]*100))
