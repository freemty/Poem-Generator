import os 
import re
import pickle
import pandas as pd 
from collections import Counter

DIR = os.path.dirname(os.path.abspath(__file__))
path = DIR + "/data/jinyong"
files = os.listdir(path)

start = 'B'
end = 'E'
texts = []
texts_str = ''

for file in files:
    with open(path + '/' + file, mode='r',encoding='gb18030') as f:
        #gb2312是内鬼嗷
        lines = f.readlines()
        for text in lines:
            if text:
                text = text[0]
                text = re.sub(pattern='[_（）《》 ]', repl='', string=text)
                texts.append(text + end)
                texts_str += text
            else:
                continue

with open(DIR + '/data/JinYong.pkl' , mode = 'wb') as f:
    pickle.dump(texts, f)

texts_len = Counter([len(i) for i in texts])
texts_len = pd.DataFrame({'length': list(texts_len.keys()), 'count': list(texts_len.values())},
                         columns=['length', 'count'])
texts_len = texts_len.sort_values(by='count', ascending=False).iloc[:10, ]
