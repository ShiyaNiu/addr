# word2vec模块
# 数据预处理模块
# 模型定义，验证集SA指标
# 模型测试
import tensorflow as tf
import numpy as np
from keras import optimizers,layers,losses,metrics
from keras.engine import data_adapter
from tensorflow_addons.layers import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
from utils.custom_metrics import costom_acc,costom_f1,costom_sa


class BiLstm(tf.keras.Model):

    def __init__(self,vocab_size,embed_dim,hidden_dim,output_dim,dropout=0.1,**kwargs):
        super(BiLstm,self).__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size,embed_dim,mask_zero=True,trainable=True)
        self.bilstm = layers.Bidirectional(
            layers.LSTM(units=hidden_dim,return_sequences=True))
        self.dense = layers.TimeDistributed(
            layers.Dense(output_dim))
        self.crf = CRF(output_dim,use_kernel=False,use_boundary=False,name="crf")
        self.dropout = layers.Dropout(dropout)

        self.optimizer=optimizers.Adam()

        self.acc_tracker = costom_acc()
        self.f1_tracker = costom_f1()
        self.sa_tracker = costom_sa()
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self,inputs):
        x = self.embedding(inputs)
        x = self.bilstm(x)
        x = self.dropout(x)
        y = self.dense(x)
        decoded_sequence, potentials, sequence_length, chain_kernel  = self.crf(y)
        return decoded_sequence
    
    def sub_mode(self,inputs):
        x = self.embedding(inputs)
        x = self.bilstm(x)
        x = self.dropout(x)
        y = self.dense(x)
        y = self.crf(y)
        return y
    
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        decoded_sequence, potentials, sequence_length, chain_kernel = self.sub_mode(x)
        crf_loss = tf.reduce_mean(-crf_log_likelihood(potentials,y,sequence_length,chain_kernel)[0])
        # mask = tf.cast(tf.not_equal(y,0),tf.float32)
        # dense_loss = losses.SparseCategoricalCrossentropy(from_logits=True)(y,potentials,mask)
        return crf_loss
    
    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        for m in self.metrics:
            if m.name == "loss":
                m.update_state(loss)
            else:
                m.update_state(y,y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        loss = self.compute_loss(x, y, y_pred, sample_weight)

        for m in self.metrics:
            if m.name == "loss":
                m.update_state(loss)
            else:
                m.update_state(y,y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    @property
    def metrics(self):
        return [self.loss_tracker,self.acc_tracker,self.f1_tracker,self.sa_tracker]


if __name__ == "__main__":
    model = BiLstm(vocab_size=2438,
                   embed_dim=100,
                   hidden_dim=200,
                   output_dim=59)
    