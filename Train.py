import logging
import time
import os
import tensorflow as tf 
from TG_Model import TGModel
from Process_Poetry import Process_Poetry
from Config import Config


DIR = os.path.dirname(os.path.abspath(__file__))

Config.data_path = DIR + '/data/Poem.pkl'
Config.model_path = DIR + '/model/train'

checkpoint_path = DIR + '/model/checkpoint'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = Process_Poetry(Config)
x_data,y_data,y_mask,vocab= data.run()




with tf.Graph().as_default():
    print("Building Model")
    start = time.time()
    model = TGModel(Config,'train')
    tensors = model.build()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config= tf.ConfigProto(gpu_options = gpu_options)) as sess:
        sess.run(init)
        if os.path.exists(checkpoint_path):
            checkpoint = tf.train.latest_checkpoint(DIR + '/model','checkpoint')
            saver.restore(sess, checkpoint)
        
        #sess.run(model.global_steps)
        model.fit(sess,saver,x_data,y_data,y_mask,vocab)








