
class Config:

    data_path = None#process     
    model_path = None
    data_type = 'poem'
    mode_type = 'length'
    len_max = 200
    len_min = 0
    vocab_size = None
    maxlen = 40 #单句最长
    one_hot = False
    cut = None
    #train
    lr = 0.01
    batch_size = 64
    hidden_size = 128#embed_size
    layer_num = 2
    n_epochs = 50
    cell = 'lstm'

    
