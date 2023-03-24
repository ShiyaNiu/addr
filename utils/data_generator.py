import tensorflow as tf
from hanlp.utils.io_util import read_tsv
import os
import json

class DataGenerator():
    def __init__(self,save_dir="addr/data/tianchi/"):
        self.text_vocab, self.label_vocab = self.load_vocabs(save_dir, filename='vocabs.json')
        self.vectorizer_text_layer = tf.keras.layers.TextVectorization(standardize=None,output_mode='int',vocabulary=self.text_vocab)
        self.vectorizer_label_layer = tf.keras.layers.TextVectorization(standardize=None,output_mode='int',vocabulary=self.label_vocab)

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        with open(os.path.join(save_dir,filename),'r',encoding="utf-8") as file:
            vocab = json.load(file)
            text_vocab = vocab["word_vocab"]["idx_to_token"]
            label_vocab = vocab["tag_vocab"]["idx_to_token"]
            return text_vocab,label_vocab
    
    def generator_data(self,trn_path):
        for sent in read_tsv(trn_path):
            words = [cells[0] for cells in sent]
            tags = [cells[1] for cells in sent]
            if not words:
                continue
            yield [" ".join(words)], [" ".join(tags)]
    
    def generator_fun(self, tsv_path):
        def generator():
            yield from self.generator_data(tsv_path)
        return generator

    def samples_to_dataset(self, file_path, batch_size=32, shuffle=None, repeat=None,
                        prefetch=1, cache=None, output_types=(tf.float32), output_shapes=([None])):
        
        samples = self.generator_fun(file_path)
        dataset = tf.data.Dataset.from_generator(samples, output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.batch(batch_size).map(
            lambda x,y: (tf.cast(self.vectorizer_text_layer(x),tf.int32),tf.cast(self.vectorizer_label_layer(y),tf.int32)))
        
        dataset = dataset.cache(cache)
        if shuffle:
            dataset = dataset.shuffle(shuffle,reshuffle_each_iteration=True)
        if repeat:
            dataset = dataset.repeat(repeat)
        if prefetch:
            dataset = dataset.prefetch(prefetch)
        # for i in dataset.take(1):
        #     print(i)
        return dataset

if __name__ == "__main__":
    batch_size = 32

    trn_path = 'addr/data/tianchi/train.conll'
    dg = DataGenerator()
    gen = dg.samples_to_dataset(trn_path, batch_size=batch_size, shuffle=10, repeat=None,
                        prefetch=5, cache='addr/utils/model-cache/train/', 
                        output_types=(tf.string, tf.string), output_shapes=([1,], [1,]))
    