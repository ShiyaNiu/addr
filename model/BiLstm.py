# word2vec模块
# 数据预处理模块
# 模型定义，验证集SA指标
# 模型测试
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,losses,optimizers
from tensorflow_addons.layers import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
from utils.custom_metrics import costom_f1,costom_sa

def custom_error(y_true, y_pred):
    _, potentials, sequence_length, chain_kernel = y_pred
    crf_loss = tf.reduce_mean(-crf_log_likelihood(potentials,y_true,sequence_length,chain_kernel)[0])
    # dense_loss = losses.SparseCategoricalCrossentropy(from_logits=True,reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)(y_true,potentials)
    return crf_loss

class BiLstm(tf.keras.Model):

    def __init__(self,vocab_size,embed_dim,hidden_dim,output_dim,dropout=0.1,**kwargs):
        super(BiLstm,self).__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size,embed_dim,mask_zero=True,trainable=True)
        self.bilstm = layers.Bidirectional(
            layers.LSTM(units=hidden_dim,return_sequences=True))
        self.dense = layers.TimeDistributed(
            layers.Dense(output_dim))
        self.crf = CRF(output_dim,use_kernel=False,use_boundary=False)
        self.dropout = layers.Dropout(dropout)

    def call(self,inputs):
        x = self.embedding(inputs)
        x = self.bilstm(x)
        x = self.dropout(x)
        y = self.dense(x)
        y = self.crf(y)
        return y


if __name__ == "__main__":
    
    #构建模型
    model = BiLstm()
    model.compile(optimizer=optimizers.Adam(), loss=custom_error)
