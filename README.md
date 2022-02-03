# Baby Poem Generate(2020.5)

## Requirement
[![](https://img.shields.io/badge/Python-3.5,3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/pandas-0.23.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.23.0)
[![](https://img.shields.io/badge/numpy-1.14.3-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.14.3)
[![](https://img.shields.io/badge/keras-2.1.6-brightgreen.svg)](https://pypi.python.org/pypi/keras/2.1.6)
[![](https://img.shields.io/badge/tensorflow-1.4,1.6-brightgreen.svg)](https://pypi.python.org/pypi/tensorflow/1.6.0)<br>

## 介绍

刚上手NLP和Tensorflow,第一个拿来练手的小项目,就是,代码借鉴了[Text_Generate](https://github.com/renjunxiang/Text_Generate),功能上就是可以实现唐诗的生成.

## 主要模块

- **Process Data**
  洗文本，切割长度，做padding，生成data batch和mask batch
- **Model**
  很简单，就是两层basicLSTM

- **Generate**
  在训练一定程度后，就可以进行生成，可以选择输入start-token或者以随机词开始，要想形成诗，还是需要依赖人为的断句约束，可以通过修改`poemGenerate.py`生成部分的约束条件来改变短句长度

## 生成样例

**《春》**
春日不能来处苦。<br>
今岁不应同梦泪，<br>
却笑花中一千载？<br>
今年春尽是何由？<br>

**《天子》**
天子龙城万丈春，黄河万国五年风。<br>
九衢无路何时日，犹得三春作帝王。<br>

欙阳宫殿里。<br>
金炉上殿，凤箫相对花中扇。<br>
春花落叶花枝里，绿叶落。<br>
春来月夜，红草斜。<br>

**《沙场》**
沙场不觉有新声，<br>
万古犹能不自伤。<br>
若到江城花满夜。<br>
莫教心绪谁同事？<br>
西山有心心断处。<br>

## 待办

- 训练的暂停与重启 done！
- mask done！
- 更大的数据集 done！
- 加入attention Todo
