import tensorflow as tf
from hanlp.utils.io_util import read_tsv
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.common.structure import SerializableDict
from hanlp.common.vocab import Vocab
import os

def load_vocabs(transform, save_dir, filename='vocabs.json'):
    vocabs = SerializableDict()
    vocabs.load_json(os.path.join(save_dir, filename))
    for key, value in vocabs.items():
        vocab = Vocab()
        vocab.copy_from(value)
        setattr(transform, key, vocab)

class DataGenerator():
    def __init__(self, transform):
        self.transform = transform

    def generator_data(self,trn_path):
        for sent in read_tsv(trn_path):
            words = [cells[0] for cells in sent]
            tags = [cells[1] for cells in sent]
            if not words:
                continue
            yield words, tags
    
    def generator_fun(self, tsv_path):
        def generator():
            yield from self.generator_data(tsv_path)
        return generator

    def samples_to_dataset(self, file_path, batch_size=32, shuffle=None, repeat=None,
                        drop_remainder=False, prefetch=1, cache=True, 
                        output_types=(tf.float32), output_shapes=([None]), padding_values=0.0):
        def data_map(words, tags):
            words = tf.cast(words, tf.string)
            tags = tf.cast(tags, tf.string)
            w = tf.cast(self.transform.x_to_idx(words), tf.int32)
            y = tf.cast(self.transform.y_to_idx(tags), tf.int32)
            return w, y
        
        samples = self.generator_fun(file_path)
        dataset = tf.data.Dataset.from_generator(samples, output_types=output_types, output_shapes=output_shapes)

        if batch_size:
            dataset = dataset.padded_batch(batch_size, ([None], [None]), padding_values, drop_remainder)
        
        dataset = dataset.map(data_map, num_parallel_calls=24)
        if cache:
            dataset = dataset.cache(cache)
        if shuffle:
            dataset = dataset.shuffle(shuffle)
        if repeat:
            dataset = dataset.repeat(repeat)
        if prefetch:
            dataset = dataset.prefetch(prefetch)
        return dataset

if __name__ == "__main__":
    trn_path = '../data/tianchi/train.conll'
    batch_size = 32
    transform = TSVTaggingTransform()
    load_vocabs(transform, "../data/tianchi/", filename='vocabs.json')
    dg = DataGenerator(transform)
    gen = dg.samples_to_dataset(trn_path, batch_size=batch_size, shuffle=None, repeat=None,
                        drop_remainder=False, prefetch=10, cache='./model-cache/train/origin', 
                        output_types=(tf.string, tf.string), output_shapes=([None], [None]), padding_values=('<pad>', '<pad>'))
    
    import time
    a = time.time()
    t = 0
    for i in gen:
        t+=1
    b = time.time()
    print(a,b)