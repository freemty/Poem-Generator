import tensorflow as tf 
import numpy as np 
import pickle as pkl
import re

import os

from TG_Model import TGModel
from Process_Poetry import Process_Poetry
from Config import Config

class generate():

    def __init__(self,config,vocab):
        self.config = Config
        self.vocab = vocab


    
    def generate_text(self,correct = True):
        
        model = TGModel(self.config , 'prediction')
        tensors = model.build()


        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(init)
            checkpoint = tf.train.latest_checkpoint(DIR + '/model/','checkpoint')
            saver.restore(sess , checkpoint)

            while True:
                print('开始生成文本(需要模型),请输入开始词,或quit以退出')
                start_word = input()
                if start_word == 'quit':
                    print('\n再见')
                    break
                if start_word == "":
                    words = list(vocab.keys())
                    for i in ['。', '？', '！', 'e']:
                        words.remove(i)
                    start_word = np.random.choice(words,1)
                try:
                    print('Start!')
                    input_index = []
                    for i in start_word:
                        index_next = vocab[i]
                        input_index.append(index_next)
                    input_index = input_index[:-1]

                    punctuation = [vocab['，'], vocab['。'], vocab['？']]
                    punctuation_index = len(start_word)
                    while index_next not in [0 , vocab['e']] :
                        input_index.append(index_next)
                        feed = {model.input_placeholder : np.array([input_index])}
                        y_predict , last_state = sess.run([tensors['prediction'],tensors['last_state']],
                                                            feed_dict = feed)

                        y_predict = y_predict[-1]
                        y_predict = {num : i for num , i in enumerate(y_predict)}
                        index_max = sorted(y_predict , reverse = True , key = lambda x: y_predict[x])[:10]
                        index_next = np.random.choice(index_max)
                        if index_next in [0 , vocab['e']] and len(input_index) < 25:
                            index_next = np.random.choice(index_max)

                        punctuation_index += 1

                        if correct:
                            if index_next in punctuation and punctuation_index < 8:
                                while index_next in punctuation:
                                    index_next = np.random.choice(index_max)
                            elif punctuation_index >= 8:
                                punctuation_index = 0
                                while (set(punctuation) & set(index_max)) and (index_next not in punctuation):
                                    index_next = np.random.choice(index_max)
                            else:
                                pass

                        if len(input_index) > self.config.maxlen:
                            break
                    int2voc = {self.vocab[word] : word for word in self.vocab}
                    text = [int2voc[i] for i in input_index]
                    text = ''.join(text)

                except Exception as e:
                        print(e)
                        text = '不能识别%s' % start_word

                finally:
                        text_list = re.findall(pattern='[^。？！]*[。？！]', string=text)
                        print('作诗完成')
                        for i in text_list:
                            print(i)

                        print('\nXD\n')

if __name__ == "__main__":


    DIR = os.path.dirname(os.path.abspath(__file__))
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    Config.data_path = DIR + '/data/Tang_Poetry.pkl'
    Config.model_path = DIR + '/model/train'

    '''
    data = Process_Poetry(Config)
    _,_,_,vocab= data.run()
    with open('data/vocab.pkl','wb')as f:
        pkl.dump(vocab ,f)
    '''
    with open('data/vocab.pkl','rb')as f:
        vocab = pkl.load(f)
    Config.vocab_size = len(vocab.keys()) + 1
    gen = generate(Config,vocab)

    gen.generate_text()

'''
天子龙城去复平，东方南去几何同。
青山不识三峰里，今去春生在水空。

金陵水边日日暮，
月华空望云云间，
行处时处无穷恨。
一片灯头满窗中？

金陵风吹雪叶中，
风烟袅杏风前起，
玉盘帘里空中醉。
秋光如落酒头回？

'''