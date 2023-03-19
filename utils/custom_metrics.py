import tensorflow as tf
from tensorflow.keras import metrics

class costom_f1(metrics.Metric):
    def __init__(self, name="f1", **kwargs):
        super(costom_f1, self).__init__(name=name, **kwargs)
        self.classes = ['prov', 'city', 'district', 'town', 'community', 'poi', 'road', 'roadno', 'otherinfo', 'subpoi', 'devzone', 'houseno', 'intersection', 'assist', 'cellno', 'floorno', 'distance', 'village_group']
        self.tp = self.init_dict_keys("tp")
        self.true_nums = self.init_dict_keys("true_nums")
        self.pred_nums = self.init_dict_keys("pred_nums")

    def update_state(self, y_true, y_pred):
        y_p, _,  _, _ = y_pred
        self.count_entity_dict( y_true, mode='1')
        self.count_entity_dict( y_p, mode='2')
        self.count_correct_entity( y_true, y_p)

    def count_entity_dict(self, y_true, mode='1'):
        flag = 0
        for seq in y_true:
            for tag in seq:
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
            ##################
            flag = 0
            for tag_t,tag_p in zip(seq_t,seq_p):
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
        p = tf.reduce_sum(self.tp.values()) / tf.reduce_sum(self.pred_nums.values())
        r = tf.reduce_sum(self.tp.values()) / tf.reduce_sum(self.true_nums.values())
        f1 = 2 * (p * r) / (p + r)
        return f1

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0.0)
        self.true_nums.assign(0.0)
        self.pred_nums.assign(0.0)

class costom_sa(metrics.Metric):
    def __init__(self, name="sa", **kwargs):
        super(costom_f1, self).__init__(name=name, **kwargs)
        self.sa = self.add_weight(name="costom_sa", initializer="zeros")
        self.all = self.add_weight(name="all", initializer="zeros")

    def update_state(self, y_true, y_pred):
        for seq_t,seq_p in zip(y_true, y_pred):
            # 对填充做处理
            ##################
            flag = 1
            for tag_t,tag_p in zip(seq_t,seq_p):
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