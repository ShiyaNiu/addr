import tensorflow as tf
from keras import metrics,losses
from keras.utils import losses_utils
import os,json,sys

tf.config.run_functions_eagerly(True)


class costom_f1(metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super(costom_f1, self).__init__(name=name, **kwargs)
        self.classes = ['prov', 'city', 'district', 'town', 'community', 'poi', 'road', 'roadno', 'otherinfo', 'subpoi', 'devzone', 'houseno', 'intersection', 'assist', 'cellno', 'floorno', 'distance', 'village_group']
        self.label_vocab = ['', '[UNK]']
        self.label_vocab.extend(self.load_vocabs())
        self.tp = self.init_dict_keys("tp")
        self.true_nums = self.init_dict_keys("true_nums")
        self.pred_nums = self.init_dict_keys("pred_nums")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_p = tf.cast(tf.argmax(y_pred,axis=-1),tf.int32)
        y_p = tf.cast(y_pred, "int32")
        y_true = tf.cast(y_true, "int32")
        self.count_entity_dict(y_true, mode='1')
        self.count_entity_dict(y_p, mode='2')
        self.count_correct_entity( y_true, y_p)
    
    def load_vocabs(self, save_dir="addr/data/tianchi/", filename='vocabs.json'):
        with open(os.path.join(save_dir,filename),'r',encoding="utf-8") as file:
            vocab = json.load(file)
            label_vocab = vocab["tag_vocab"]["idx_to_token"]
            return label_vocab

    def count_entity_dict(self, y_true, mode='1'):
        flag = 0
        for seq in y_true:
            for tag in seq:
                tag = tag.numpy()
                if tag == 0:#填充部分
                    break
                tag = self.label_vocab[tag]
                if 'B-' in tag and flag == 0:
                    flag = 1
                if ('E-' in tag and flag == 1) and 'S-' in tag:
                    flag = 0
                    tag_name = tag[2:]
                    if mode == '1':
                        self.true_nums[tag_name].assign_add(1.0)
                    if mode == '2':
                        self.pred_nums[tag_name].assign_add(1.0)
                if tag == 'O':
                    flag == 0

    def count_correct_entity(self, y_true, y_pred):
        # 'BIESO' 'B-,S,O'
        for seq_t,seq_p in zip(y_true, y_pred):
            # 建立索引和tag的映射
            # 对填充做处理
            flag = 0
            for tag_t,tag_p in zip(seq_t,seq_p):
                tag_t,tag_p = tag_t.numpy(),tag_p.numpy()
                if tag_t == 0 or tag_p == 0:#填充部分
                    break
                tag_t,tag_p = self.label_vocab[tag_t],self.label_vocab[tag_p]
                if 'B-' in tag_t and tag_t == tag_p and flag == 0:
                    flag = 1
                if 'I-' in tag_t and tag_t == tag_p and flag == 1:
                    flag = 1
                if 'E-' in tag_t and tag_t == tag_p and flag == 1:
                    flag = 0
                    tag = tag_t[2:]
                    self.tp[tag].assign_add(1.0)
                if 'S-' in tag_t and tag_t == tag_p and flag == 0:
                    tag = tag_t[2:]
                    self.tp[tag].assign_add(1.0)
                if tag_t != tag_p or tag_t == 'O':
                    flag = 0
                
    def init_dict_keys(self, name):
        dic = {}
        for cls in self.classes:
            dic[cls] = self.add_weight(name="{}_{}".format(name,cls), initializer="zeros")
        return dic

    def result(self):
        f1 = 0.0
        tp = tf.reduce_sum(list(map(lambda x:x.numpy(),list(self.tp.values()))))
        pred_nums = tf.reduce_sum(list(map(lambda x:x.numpy(),list(self.pred_nums.values()))))
        true_nums = tf.reduce_sum(list(map(lambda x:x.numpy(),list(self.true_nums.values()))))
        if pred_nums.numpy() and true_nums.numpy():
            p = tp / pred_nums
            r = tp / true_nums
            f1 = 2 * (p * r) / (p + r)
            f1 = f1.numpy()
        return max(0.0,f1)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        for key in self.classes:
            self.tp[key].assign(0.0)
            self.true_nums[key].assign(0.0)
            self.pred_nums[key].assign(0.0)

class costom_sa(metrics.Metric):
    def __init__(self, name="sa", **kwargs):
        super(costom_sa, self).__init__(name=name, **kwargs)
        self.sa = self.add_weight(name="costom_sa", initializer="zeros")
        self.all = self.add_weight(name="all", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_p = tf.cast(tf.argmax(y_pred,axis=-1),tf.int32)
        y_p = tf.cast(y_pred, "int32")
        y_true = tf.cast(y_true, "int32")
        for seq_t,seq_p in zip(y_true, y_p):
            flag = 1
            for tag_t,tag_p in zip(seq_t,seq_p):
                tag_t,tag_p = tag_t.numpy(),tag_p.numpy()
                # 对填充做处理
                if tag_t == 0 and tag_p == 0:
                    break
                elif tag_t == 0:
                    flag = 0
                    break
                elif tag_p == 0:
                    flag = 0
                    break
                ##################
                if tag_t == tag_p:
                    continue
                else:
                    flag = 0
                    break
            if flag == 1:
                self.sa.assign_add(1.0)
            self.all.assign_add(1.0)

    def result(self):
        return tf.divide(self.sa,self.all)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sa.assign(0.0)
        self.all.assign(0.0)

# if __name__ == "__main__":
#     print(tf.executing_eagerly())
#     print(tf.config.experimental.list_physical_devices("CPU"))
#     print(tf.config.experimental.list_physical_devices("GPU"))