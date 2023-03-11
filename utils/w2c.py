from hanlp.utils.io_util import read_tsv_as_sents
import random
import os
import json
import time
import logging
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

class Data2Gensim:
    """
    格式化输入数据，符合GENSIM模型要求
    """
    def __init__(self, save_path=r'D:\daydayup\NLP\addr\data\tianchi\gensim_data.txt'):
        # output format data
        self.gensim_data = []
        # show the input and output data
        self.show_nums = 1
        self.count = 0
        self.gensim_data_num = 0
        # 保存路径
        self.save_path = save_path

    def char2word(self,sent):
        '''
        返回数据：浙江,杭州市,江干区,九堡镇,三村村,一区|prov,city,district,town,community,poi
        '''
        chars = [cells[0] for cells in sent]
        tags = [cells[1] for cells in sent]
        if not chars:
            return
        words = []
        labels = []
        word = ''
        for i in range(len(chars)):
            ch = chars[i]
            tag = tags[i]
            if tag == 'O':
                words.append(ch)
                labels.append(tag)
                word = ''
                continue
            tag_left,tag_right = tag.strip().split('-')
            if tag_left == 'B':
                word = ch
                labels.append(tag_right)
            elif tag_left == 'I':
                word += ch
            elif tag_left == 'E':
                word += ch
                words.append(word)
            elif tag_left == 'S':
                words.append(ch)
                labels.append(tag_right)
                word = ''
        assert len(words)==len(labels),"ERROR {} {}".format(words,labels)
        return words,labels

    def format_to_gensim(self,path):
        """
        """
        for addr in read_tsv_as_sents(path):
            words,labels = self.char2word(addr)
            if self.count<self.show_nums:
                    print("origin data:", addr)
            if self.count<self.show_nums:
                    self.count += 1
                    print("format data:",words)
            self.gensim_data.append(words)
            self.gensim_data_num += 1

        random.shuffle(self.gensim_data)
        print("---------------format_to_gensim_processing completed-----------------,totoal data nums:{}".format(self.gensim_data_num))

    def save_format_data(self):
        """
        将处理过得训练数据保存到self.save_path的指定文件
        """
        if self.gensim_data:
            if os.path.exists(self.save_path):
                print("-----------file name has already exists, don't save-----------")
            else:
                with open(self.save_path,'w',encoding='utf-8') as file:
                    json.dump(
                                self.gensim_data,
                                file,
                                indent=4,
                                ensure_ascii=False
                            )
                    print("-----------format data saving completed-----------")
        else:
            print("-----------no data to save-----------")

# callback
class EpochSaver(CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''

    def __init__(self, save_dir, save_name, logger):
        self.save_path = save_dir + save_name
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        self.logger = logger
    
    def on_train_begin(self, model):
        self.logger.info("-------------------word2vec,cws source:TEXT from train data--------------------")
        self.logger.info("parameters:vector_size=100,window=5,min_count=10,workers=4,sg=0,hs=0,negative=10,epochs=2000")

    def on_epoch_end(self, model):
        self.epoch += 1
        cur_loss = model.get_latest_training_loss() #epochs累加
        epoch_loss = cur_loss - self.pre_loss
        time_taken = time.time() - self.since
        if self.epoch % 100 == 0:
            self.logger.info("Epoch %d, loss: %.2f, time: %dmin %ds" %
                (self.epoch, epoch_loss, time_taken // 60, time_taken % 60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            self.logger.info("Better model in epoch %d... Best loss: %.2f" % (self.epoch, self.best_loss))
            model.save(self.save_path)
        self.pre_loss = cur_loss
        self.since = time.time()

def print_logs(name,file_path):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

if __name__ == "__main__":

    # 生成符合GENSIM格式的输入数据
    # train_path = r"D:\daydayup\NLP\addr\data\tianchi\train.conll"
    # dev_path = r"D:\daydayup\NLP\addr\data\tianchi\dev.conll"
    # data = Data2Gensim()
    # data.format_to_gensim(train_path)
    # data.format_to_gensim(dev_path)
    # data.save_format_data()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 训练数据加载
    train_data_path = r'D:\daydayup\NLP\addr\data\tianchi\gensim_data.txt'
    save_dir = './model_dir/'
    save_name = "w2v"+".model"
    log_save_dir = "./logs/"
    log_save_name = "log"+".log"
    logger_name = "w2v_train_log"

    with open(train_data_path, 'r', encoding='utf-8') as train:
        train_data_gensim = json.load(train)
    print("size: %d . data load completed..." % len(train_data_gensim))

    # 打印日志
    logger = print_logs(logger_name,log_save_dir+log_save_name)

    # train the word2vec model
    since = time.time()
    model = Word2Vec(sentences=train_data_gensim,vector_size=100,window=5,min_count=1,workers=4,sg=0,hs=0,negative=10,
                    epochs=2000,compute_loss=True,callbacks=[EpochSaver(save_dir,save_name,logger)])
    time_elapsed = time.time() - since
    print('Time to train: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))