from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import RepeatVector



def keras_lstm(max_sequence_length, vectors_vocabulary_size, vectors_dimension=300, 
               pre_trained_vectors=False, tensorflow_gpu=False):
    
    if pre_trained_vectors is True:
        '''
        print("USE PRE TRAINED")
        num_words = min(max_fatures, len(word_index) + 1)
        weights_embedding_matrix = load_pre_trained_wv(word_index, num_words, word_embedding_dim)
        input_shape = (max_sequence_length,)
        model_input = Input(shape=input_shape, name="input", dtype='int32')    
        embedding = Embedding(
            num_words, 
            word_embedding_dim,
            input_length=max_sequence_length, 
            name="embedding", 
            weights=[weights_embedding_matrix], 
            trainable=False)(model_input)
        if bilstm is True:
            lstm = Bidirectional(LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)
        else:
            lstm = LSTM(word_embedding_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)
        '''

    else:
        # Faz uma tupla que define o tamanho da entrada do modelo:
        inputs_shape = (max_sequence_length,)

        # ENCODER:
        # Define a primeira camada a qual recebe as entradas do modelo:
        model_input = Input(shape=inputs_shape, name="input", dtype='int32')    

        # Aqui é aplicado o Embedding do próprio Keras para transformar as palavras em vetores baseados na suas posições e
        # frequência ao longo do texto (o qual estará organizado em sequeência de tokens)
        model_embedding = Embedding(vectors_vocabulary_size, vectors_dimension, 
                                    input_length=vectors_vocabulary_size, name="model_embedding")(model_input)

        if tensorflow_gpu is True:
            # Define o formato de uma célula LSTM que será utilizada por palavra: 
            encoder_lstm = CuDNNLSTM(vectors_dimension, dropout=0.2, recurrent_dropout=0.2, name="encoder_lstm")(model_embedding)
        else:
            # Define o formato de uma célula LSTM que será utilizada por palavra: 
            encoder_lstm = LSTM(vectors_dimension, dropout=0.2, recurrent_dropout=0.2, name="encoder_lstm")(model_embedding)

        # Repete o processo anterior um número de vezes igual ao tamanho da sequência de palavras, ou seja, para cada palavra
        # faz isso uma vez
        encoder = RepeatVector(max_sequence_length, name="encoder")(encoder_lstm)

        # DECODER:
        decoder_input = Input(shape=inputs_shape, name="decoder_input", dtype='int32')
        decoder_embedding = Embedding(vectors_vocabulary_size, vectors_dimension, name="decoder_embedding")(decoder_input)
        decoder_concatenate = concatenate([encoder, decoder_embedding])

        if tensorflow_gpu is True: 
            decoder_lstm = CuDNNLSTM(vectors_dimension, dropout=0.2, recurrent_dropout=0.2, name="decoder_lstm")(decoder_concatenate)
        else:
            decoder_lstm = LSTM(vectors_dimension, dropout=0.2, recurrent_dropout=0.2, name="decoder_lstm")(decoder_concatenate)

        model_output = Dense(vectors_vocabulary_size, activation='softmax')(decoder_lstm)

    model = Model(inputs=[model_input, decoder_input], outputs=model_output)

    return model
