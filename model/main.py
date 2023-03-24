import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import tensorflow as tf
from keras import optimizers,losses
from utils.custom_metrics import costom_f1,costom_sa
from utils.data_generator import DataGenerator
from BiLstm import BiLstm

if __name__ == "__main__":
    batch_size = 64
    # 训练数据读取
    trn_path = 'addr/data/tianchi/train.conll'
    dg_trn = DataGenerator()
    trn_data = dg_trn.samples_to_dataset(trn_path, batch_size=batch_size, shuffle=10, repeat=None,
                        prefetch=5, cache='addr/utils/model-cache/train/', 
                        output_types=(tf.string, tf.string), output_shapes=([1,], [1,]))
    dev_path = 'addr/data/tianchi/dev.conll'
    dg_dev = DataGenerator()
    dev_data = dg_dev.samples_to_dataset(dev_path, batch_size=batch_size, shuffle=10, repeat=None,
                        prefetch=5, cache='addr/utils/model-cache/dev/', 
                        output_types=(tf.string, tf.string), output_shapes=([1,], [1,]))

    #构建模型
    model = BiLstm(vocab_size=dg_trn.vectorizer_text_layer.vocabulary_size(),
                   embed_dim=100,
                   hidden_dim=200,
                   output_dim=dg_trn.vectorizer_label_layer.vocabulary_size())
    #模型编译
    model.compile(optimizer=optimizers.Adam(), 
                  metrics=['accuracy',costom_f1(),costom_sa()])
    #模型训练
    csv_logger = tf.keras.callbacks.CSVLogger('addr/logs/bilstm_train_0320.log')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=20,
        verbose=1,
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="addr/model_dir/bilstm_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
    model.fit(trn_data,
              epochs=2000,
              callbacks=[csv_logger,early_stop,model_checkpoint],
              validation_data=dev_data)
    print(model.summary())
    