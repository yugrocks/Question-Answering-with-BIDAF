from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply,TimeDistributed
from keras.layers import RepeatVector, Dense, Activation, Lambda,Reshape
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model


query_len = 50 # J
context_len =  300 # T
embed_len = 50 # d


contextual_lstm = Bidirectional(LSTM(embed_len, return_sequences=True, name="bidir_lstm_contextual"))
contextual_lstm2 = Bidirectional(LSTM(embed_len, return_sequences=True, name="bidir_lstm_contextual2"))
alpha_weights = Dense(1, activation="linear",name="alpha_weights")
concatenator = Concatenate(axis=-2,name= "concatenator")
modeling_layer1 = Bidirectional(LSTM(embed_len, return_sequences=True, name="bidir_lstm_contextual"))
modeling_layer2 = Bidirectional(LSTM(embed_len, return_sequences=True, name="bidir_lstm_contextual"))



def get_S_matrix_alternate(k):
    contextual_embed_query, contextual_embed_context = k
    duplicated_ceq = K.concatenate([contextual_embed_context]*query_len,axis=1) # shape None,TJ, 2d
    tensors = []
    for i in range(query_len):
        tensors.append(K.repeat(contextual_embed_query[:,i,:],context_len))
    tensors = K.concatenate(tensors,axis=1) # shape None,TJ, 2d
    mult = Multiply()([duplicated_ceq, tensors])
    tensors = K.concatenate([duplicated_ceq,tensors,mult],axis=-1) # shape = JT x 6d
    return tensors
    # now multiply each row by the same weight vector
    #tensors = TimeDistributed(alpha_weights) (tensors)
    # now reshape tensors to J x T
    #tensors = Reshape((query_len, context_len))(tensors) # shape of S = J x T
    #return tensors # tensors is S itself


def context_to_query_attention(k):
    # T vectors of length J should be there. a_t is simply the J dimension wise softmax of the S
    S,contextual_embed_query,contextual_embed_context = k
    a = K.softmax(S, axis=1)
    contextual_embed_query_trans = Permute((2,1))(contextual_embed_query)
    U_bar = K.batch_dot(contextual_embed_query_trans, a) # shape would be 2d x T
    return U_bar
    
    
def query_to_context_attention(k):
    S, contextual_embed_context = k
    b = K.max(S , axis=1) # a vector of shape None x T
    multiples = []
    for t in range(context_len):
        multiplication = contextual_embed_context[:,t,:] * K.expand_dims(b[:,t],axis=-1)
        multiples.append(multiplication)
    multiples2 = K.stack(multiples, axis=1)
    h_bar = K.sum(multiples2, axis=1)
    H_bar =  K.stack([h_bar,] *context_len, axis=-1)# stack h_bar T times 
    return H_bar
    

def megamerge(k):
    U_bar, H_bar, contextual_embed_context = k
    contextual_embed_context_trans  = Permute((2,1))(contextual_embed_context)
    H_U_bar = Multiply()([contextual_embed_context_trans,U_bar])
    H_H_bar = Multiply()([contextual_embed_context_trans,H_bar])
    concat = K.concatenate([contextual_embed_context_trans,U_bar,H_U_bar,H_H_bar], axis=1)
    return concat
    
    

def model():
    embeddings_query = Input(shape=(query_len,embed_len),name="query_embed")
    embeddings_context = Input(shape=(context_len,embed_len),name="context_embed")
    
    contextual_embed_query = contextual_lstm(embeddings_query) # returns batch_size x J x 2D
    contextual_embed_context = contextual_lstm2(embeddings_context) # returns batch_size x T x 2d
    #S = get_s_matrix(contextual_embed_query, contextual_embed_context)
    S_interm = Lambda(get_S_matrix_alternate,name="S_calculation")([contextual_embed_query, contextual_embed_context]) # shape J x T
    S_interm2 = TimeDistributed(alpha_weights) (S_interm)
    S = Reshape((query_len, context_len))(S_interm2)
    U_bar = Lambda(context_to_query_attention,name="U_bar_calc")([S,contextual_embed_query,contextual_embed_context])
    H_bar = Lambda(query_to_context_attention,name="H_bar_calc")([S, contextual_embed_context])
    # now the three matrices U_bar, H_bar, contextual_embed_context have the SAME dimension
    G = Lambda(megamerge,name="megamerge")([U_bar, H_bar, contextual_embed_context]) # returns shape None, 8d, T
    # now G goes through another Bi LSTM 
    G_trans = Permute((2,1))(G)
    M1 = modeling_layer1(G_trans) # returns T x 2d
    M2 = modeling_layer2(M1) # returns T x 2d
    # now concatenating G with M1 and M2 each
    G_M1 = Lambda(lambda x: K.concatenate(x, axis=-1), name="G_M1_calc")([G_trans, M1])# shape = None, T , 10d
    G_M2 = Lambda(lambda x : K.concatenate(x, axis=-1),name="G_M2_calc")([G_trans, M2]) # shape = None, T , 10d
    p1 = TimeDistributed(Dense(1,activation="linear"))(G_M1) # note that weights are shared. output shape = None,t, 1
    p2 = TimeDistributed(Dense(1, activation="linear"))(G_M2)
    p1 = Lambda(lambda x: K.squeeze(K.softmax(x,axis=1),axis=-1),name="start_probability_dist")(p1)
    p2 = Lambda(lambda x: K.squeeze(K.softmax(x,axis=1),axis=-1),name="end_probability_dist")(p2)
    
    model = Model(inputs=[embeddings_query, embeddings_context] , outputs=[p1,p2])
    model.compile(loss="categorical_crossentropy",optimizer=Adam(0.001),metrics=["accuracy"])
    model.load_weights("model2_weights.h5")
    return model
 

