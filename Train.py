import logging
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf 
from TG_Model import TGModel
from Process_Poetry import Process_Poetry
from Config import Config



DIR = os.path.dirname(os.path.abspath(__file__))

Config.data_path = DIR + '/data/Tang_Poetry.pkl'
Config.model_path = DIR + '/model/train'

checkpoint_path = DIR + '/model/checkpoint'

data = Process_Poetry(Config)
x_data,y_data,y_mask,vocab= data.run()


with tf.Graph().as_default():
    print("Building Model")
    start = time.time()
    model = TGModel(Config,'train')
    tensors = model.build()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(init)
        if os.path.exists(checkpoint_path):
            checkpoint = tf.train.latest_checkpoint(DIR + '/model','checkpoint')
            saver.restore(sess, checkpoint)
        
        #sess.run(model.global_steps)
        model.fit(sess,saver,x_data,y_data,y_mask,vocab)








