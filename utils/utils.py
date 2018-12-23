# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : utils.py
# Time    : 2018/11/27 0027 下午 3:32
"""

import numpy as np
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import re



def read_file_seg_sparse(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = []
        label = []
        for line in f.readlines():
            label.append(re.split('=', line.split('\t')[0])[1])
            content.append([i for i in line.split('\t')[1].strip('\n').split(' ')])

    labels = []
    for i in label:
        if int(i) == 0:
            labels.append([0, 0])
        else:
            labels.append([0, 1])
    return content, labels


def read_file_seg(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = []
        label = []
        for line in f.readlines():
            label.append(re.split('=', line.split('\t')[0])[1])
            content.append([i for i in line.split('\t')[1].strip('\n').split(' ')])
    return content, label



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def create_dictionaries(maxlen,word_vec=None, combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (word_vec is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(word_vec.keys(), allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: word_vec[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量
        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')



#加载词向量
def load_word2vec(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = {} #忽略第一行的行数和列数f.readlines()[1:]
        for line in f.readlines()[1:]:
            d[line.split(' ')[0]] = np.array([float(i) for i in line.strip('\n').split(' ')[1:]])
    return d

#获取词向量和索引
def get_data(index_dict, word_vectors,vocab_dim):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    return n_symbols, embedding_weights





