import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf 
import numpy as np


class TGModel(object):

    def fit(self,sess,saver,x_data,y_data,vocab):
        for epoch in range(self.config.n_epochs):
            losses = []
            print("Epoch {} out of {}".format(epoch + 1, self.config.n_epochs))
            #prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
            for batch in range(400):
                index_all = np.arange(len(x_data))
                index_batch = np.random.choice(index_all,self.config.batch_size)
                inputs_batch = x_data[index_batch]
                labels_batch = y_data[index_batch]
                #labels_batch = self.create_ont_hot(labels_batch.reshape([-1]))

                feed = {self.input_placeholder : inputs_batch ,\
                        self.labels_placeholder : labels_batch}
                loss,_,_ = sess.run([self.tensors['total_loss'],
                                    self.tensors['train_op'],
                                    self.tensors['last_state']],
                        feed_dict = feed)
                losses.append(loss)
                if (batch+1) % 100 == 0:
                        print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch + 1, batch + 1, losses[batch]))
                #print(sess.run(self.global_steps))

            if epoch % 1 == 0:
                saver.save(sess, self.config.model_path, global_step=self.global_steps)
                print('global_step{} , loss = {}'.format(sess.run(self.global_steps),np.sum(losses)/400))
                #prog.update(i + 1, [("train loss", loss)])
    

    def create_ont_hot(self,ids):
        one_hot = np.zeros([len(ids),self.config.vocab_size])
        for i , label in enumerate(ids):
            one_hot[i,label] = 1
        return one_hot

    def build(self):
       
        #add_placeholder
        self.input_placeholder = tf.placeholder(tf.int32 ,[None,None])
        
        self.labels_placeholder = tf.placeholder(tf.int32 ,[None,None])

        #add_embeddings
        embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size,self.config.hidden_size],-1.0,1.0))
        x = tf.nn.embedding_lookup(embeddings ,self.input_placeholder)

        #add_hidden_layers
        if self.config.cell == "gru":
            cell = tf.nn.rnn_cell.RNNCell
        elif self.config.cell == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        cell_list = [cell(self.config.hidden_size,state_is_tuple=True) for i in range(self.config.layer_num)]
        cell_mul = tf.nn.rnn_cell.MultiRNNCell(cell_list,state_is_tuple=True)

        if self.action == 'train':
            initial_state = cell_mul.zero_state(self.config.batch_size , dtype = tf.float32)
        else:
            initial_state = cell_mul.zero_state(1 , dtype = tf.float32)


        self.global_steps = tf.Variable(0,name='global_steps',trainable=False)
        #add_pred_op
        W = tf.get_variable(name = 'W' , shape = [self.config.hidden_size , self.config.vocab_size], \
            initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name = 'b' , shape = self.config.vocab_size ,\
            initializer = tf.constant_initializer(0))
            
        output , last_state = tf.nn.dynamic_rnn(cell_mul , x , initial_state = initial_state)
        output = tf.reshape(output , [-1,self.config.hidden_size])
        preds = tf.matmul(output, W) + b

        if self.action == 'train':
            #add_loss_op
            onehot_labels = tf.one_hot(tf.reshape(self.labels_placeholder , [-1]),                                         depth=self.config.vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels= onehot_labels , logits = preds,name = 'loss')
            total_loss = tf.reduce_mean(loss)

            #add_trainop
            train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss,global_step=self.global_steps)

            self.tensors['initial_state'] = initial_state
            self.tensors['last_state'] = last_state
            self.tensors['output'] = output
            self.tensors['loss'] = loss
            self.tensors['total_loss'] = total_loss
            self.tensors['train_op'] = train_op
        
        elif self.action == 'prediction':
            prediction = tf.nn.softmax(preds)
            self.tensors['prediction'] = prediction
            self.tensors['initial_state'] = initial_state
            self.tensors['last_state'] = last_state
        else:
            raise ValueError('Undefined action!')
            
        return self.tensors

    def __init__(self,config,action):
        self.action = action
        self.input_placeholder = None
        self.labels_placeholder = None
        self.config = config
        self.tensors = {}