# -*- coding: utf-8 -*-

"""
Author: kingming

File: lstm_torch.py

Time: 2019/4/15 下午7:42

License: (C) Copyright 2018, xxx Corporation Limited.

"""

from utils import utils
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable



path = "./data/test.txt,./data/test.txt"
class_weight = 0
w2v = "./word_vector/word2vec.words"
sequence_length = 80
embed_size = 300



def preprocess():
    # Data Preparation
    # ==================================================
    # Load data
    path_data = [i for i in path.split(',')]
    print("Loading data...")
    if class_weight == 0:
        x_train, y_train = utils.read_file_seg(path_data[0])
    else:
        x_train, y_train = utils.read_file_seg_sparse(path_data[0])
    # Build vocabulary
    max_document_length = max([len(x) for x in x_train])
    print(max_document_length)
    print('Loading a Word2vec model...')
    word_2vec = utils.load_word2vec(w2v)  # 加载词向量
    maxlen = sequence_length
    index_dict, word_vectors, x = utils.create_dictionaries(maxlen, word_2vec, x_train)
    print('Embedding weights...')
    vocab_size = embed_size
    word_size, embedding_weights = utils.get_data(index_dict, word_vectors, vocab_dim=vocab_size)
    # test set
    print('Test set ....')
    if class_weight == 0:
        x_test, ytest = utils.read_file_seg(path_data[1])
    else:
        x_test, ytest = utils.read_file_seg_sparse(path_data[1])
    index_dict1, word_vectors1, x_test = utils.create_dictionaries(maxlen, word_2vec, x_test)
    train_x = np.array(x)
    train_y = np.array(y_train)
    test_x = np.array(x_test)
    test_y = np.array(ytest)

    print('train_x_y_shape', train_x.shape, train_y.shape)
    print('test_x_y_shape', test_x.shape, test_y.shape)
    print("Vocabulary Size: {:d}".format(word_size))
    return train_x, train_y, test_x, test_y, embedding_weights









class RNN(nn.Module):

    def __init__(self, x, weight, embedding_dim, hidden_dim):
        """
        """
        super(RNN, self).__init__()
        weights = Variable(torch.from_numpy(weight))
        # [0-10001] => [100]
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.embedding.weight.requires_grad = False
        # [100] => [256]
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,batch_first=True,
                           bidirectional=True, dropout=0.5)
        # [256*2] => [1]
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

        self.x = x
    def forward(self):
        """
        x: [seq_len, b] vs [b, 3, 28, 28]
        """
        # [seq, b, 1] => [seq, b, 100]
        embedding = self.dropout(self.embedding(self.x))

        #permute是更灵活的transpose，可以灵活的对原数据的维度进行调换，而数据本身不变。
        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_di]
        output, (hidden, cell) = self.encoder(embedding)

        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out


x_train, y_train, x_dev, y_dev, embedding_weights, = preprocess()



rnn = RNN(x_train, embedding_weights, 300, 256)
optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
criteon = nn.BCEWithLogitsLoss()



def binary_acc(preds, y):
    """
    get accuracy
    """
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(rnn, tr_x, tr_y,optimizer, criteon):
    avg_acc = []
    rnn.train()

    for i in range(2):

        # [seq, b] => [b, 1] => [b]
        pred = rnn()
        #
        loss = criteon(pred, tr_y)
        acc = binary_acc(pred, tr_y).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(i, acc)

    avg_acc = np.array(avg_acc).mean()
    print('avg acc:', avg_acc)

train(rnn, x_train, y_train, optimizer, criteon)