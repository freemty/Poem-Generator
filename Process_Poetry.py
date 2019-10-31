import os
import time
import re
import pickle
from Config import Config

import numpy as np 
import pandas as pd 
import jieba
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Process_Poetry():
    
    def __init__(self,Config):
        self.config = Config 
        self.texts = None
        self.vocab = None
        self.x_batch = None
        self.y_batch = None
        self.y_batch_one_hot = None

    def load_data(self):
        if self.config.data_type == 'poem':
            with open(self.config.data_path ,mode = 'rb') as f:
                texts = pickle.load(f)
            texts = [i for i in texts if len(i) > self.config.len_min and len(i) < self.config.len_max]
        else:
            raise ValueError('file should be poem so far')
        #To be continue XD
        self.texts = texts
     
    def text2seq(self):
        #分词
        if self.config.data_type == 'poem':
            self.config.cut = False
        if self.config.cut:
            texts = [jieba.lcut(text) for text in self.texts]
            print('cut done!')
            tokenizer = Tokenizer(self.config.vocab_size, char_level= False)
        else:
            tokenizer = Tokenizer(self.config.vocab_size, char_level=True)
        #文本切分       
        if self.config.mode_type == 'length':
            texts_new = []
            for i in self.texts:
                mod = len(i) % self.config.maxlen
                i += ('E' *(self.config.maxlen - mod))
                for j in range(len(i) // self.config.maxlen + 1):
                    texts_new.append(i[j * self.config.maxlen:\
                        (j * self.config.maxlen + self.config.maxlen)])
        else:
            raise ValueError('mode should be length so far')
        #生成字典
        tokenizer.fit_on_texts(self.texts)
        self.vocab = tokenizer.word_index
        self.config.vocab_size = len(self.vocab.keys()) + 1
        #文本编码
        encode = tokenizer.texts_to_sequences(self.texts)
        self.encode_texts = encode
        del self.texts

    def create_batches(self):
        x_batch = []
        y_batch = []
        
        for i in self.encode_texts:
            x_batch.append(i[:-1])
            y_batch.append(i[1:])

        n,pad_seq = 0,[]
        while n < len(self.encode_texts):
            pad_seq += list(pad_sequences(x_batch[n : n+5000] , maxlen = self.config.maxlen,\
                                            padding = 'post' , value = 0 , dtype = 'int'))
        n += 5000

        self.x_batch = np.array([i[:-1] for i in pad_seq])
        self.y_batch = np.array([i[1:] for i in pad_seq])

        if self.config.one_hot:
            y_one_hot = [self.create_one_hot(i,self.config.vocab_size) for i in self.y_batch]
            self.y_batch_one_hot = y_one_hot

    def create_one_hot(self,ids,vocab_size):
        one_hot = np.zeros([len(ids),vocab_size])
        for i,id in enumerate(ids):
            one_hot[i,id] = 1
        return one_hot

    def run(self):

        self.load_data()
        self.text2seq()
        self.create_batches()
        self.x_batch = np.array(self.x_batch)

        if self.config.one_hot:
            self.y_batch = np.array(self.y_batch_one_hot)
        else:
            self.y_batch = np.array(self.y_batch)

        return self.x_batch , self.y_batch , self.vocab




