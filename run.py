import pickle
import re
import os
from Config import Config

from Process_Poetry import Process_Poetry


DIR = os.path.dirname(os.path.abspath(__file__))

Config.data_path = DIR + '/data/Tang_Poetry.pkl'
Config.model_path = DIR + '/model/train'

data = Process_Poetry(Config)
_,_,vocab = data.run()

with open('data/vocab','w') as f:
    pickle.dump(vocab, f , pickle.HIGHEST_PROTOCOL)
