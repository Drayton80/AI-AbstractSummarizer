from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import CuDNNLSTM

def lstm_keras_seq2seq(max_sequence_length=300, embedding_dimension=300):
    input_shape = (max_sequence_length,)
    model_input = Input(shape=input_shape, name="input", dtype='int32')    

    embedding = Embedding(25000, embedding_dimension, input_length=max_sequence_length, name="embedding")(model_input)

    encoder = CuDNNLSTM(embedding_dimension, dropout=0.2, recurrent_dropout=0.2, name="lstm_encoder")(embedding)

    decoder = CuDNNLSTM(embedding_dimension, dropout=0.2, recurrent_dropout=0.2, name="lstm_encoder")(encoder)
    
    model_output = Dense(2, activation='softmax', name="softmax")(decoder)

    model = Model(inputs=model_input, outputs=model_output)

    return model