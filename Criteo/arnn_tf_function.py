# this is a change from previous script as we want full functionality of tensorflow
from tensorflow.keras.layers import Embedding, Dense, Activation, concatenate, Layer,  Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.models import save_model, load_model
# from tensorflow.keras.callbacks import ModelCheckpoint, Callback
# from keras.callbacks import CSVLogger
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import backend as K


import sys
sys.path.insert(0, 'C:/Users/hp/Downloads/MTP/web_app/Criteo')


from sklearn.metrics import *
from datetime import date
import pickle

from arnn_config import config1
#load it
with open(f'config_class_arnn_criteo.pickle', 'rb') as file2:
    Config = pickle.load(file2)




# --------------------------------------------------- Attention Layer  --------------------------------------------------------------

# ## CustomAttention with Xmi for all the samples instead of last hiddent state
#weight initializers 

randNorm = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
#randNorm  = "glorot_uniform"
#randNorm_e = "uniform"
randNorm_e = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)


class CustomAttention(Layer):
    def __init__(self, units,name=None, **kwargs):
        super(CustomAttention, self).__init__(name=name,**kwargs)
        self.units=units
        self.W1 = self.add_weight(name='weight_h', shape=([Config.n_hidden,Config.n_hidden]), trainable=True)
        self.W2 = self.add_weight(name='weight_x', shape=([Config.n_input, Config.n_hidden]), trainable=True)
        self.V = self.add_weight(name='v_a', shape=([Config.n_hidden]), initializer = randNorm, trainable=True)
        super(CustomAttention, self).__init__(name=name,**kwargs)

    def call(self, query, values):
        # query : [batch_size, embedding_size*10+2 = 2562]       #hidden_size if ht is used instead of xmi
        # values: [batch_size, maxlen, hidden_size]
        # (batch_size, maxlen, units) = (batch_size, units) + (batch_size, units) 
        
        l2_regularizer = l2(Config.miu)
        self.add_loss(l2_regularizer(self.W1))
        self.add_loss(l2_regularizer(self.W2))
        self.add_loss(l2_regularizer(self.V))
        
        Ux = tf.matmul(query, self.W2)    
        e = []
        for i in range(Config.seq_max_len):
            value = values[:,i,:]                                 #(batch_size, units = n_hidden)
            score = tf.multiply(tf.tanh(tf.matmul(value, self.W1) + Ux), self.V)     # (batch_size, units=n_hidden)
            score = tf.reduce_sum(score,axis=1,keepdims=False)      # (batch_size, 1)
            e.append(score)
           
        score = tf.stack(e)                                             # (maxlen, batch_size,  1)    
        # softmax over all time_stamps
        attention_weights = tf.nn.softmax(score, axis=0)                # ( maxlen, batch_size, 1)
        attention_weights = tf.transpose(attention_weights,[1,0])     # ( batch_size, maxlen, 1)
        attention_weights = tf.expand_dims(attention_weights, -1) 
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
    def get_config(self):
        config = super(CustomAttention, self).get_config()
        config.update({
            'units':self.units,'name':self.name})
        return config
    
    
class XMI_lay(Layer):
    def __init__(self,name=None, **kwargs):
        super(XMI_lay, self).__init__(name=name,**kwargs)
        super(XMI_lay, self).__init__(name=name,**kwargs)
        
    def call(self, seqlen, x):
        batch_s = K.shape(x)[0]
        seqlen = tf.reshape(seqlen,[-1])
        seqlen =tf.dtypes.cast(seqlen, tf.int32)
        index = tf.range(0, batch_s) * Config.seq_max_len + (seqlen - 1)
        x_last = tf.gather(tf.reshape(x, [-1, Config.n_input]), index)
        return x_last
    
    def get_config(self):
        config = super(XMI_lay, self).get_config()
        config.update({
            'name':self.name})
        return config