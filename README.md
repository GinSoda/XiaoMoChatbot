## 项目名：小墨Chatbot

### 项目简介：

​		借鉴了[K-BERT](https://github.com/autoliuweijie/K-BERT)、[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)、[WRD](https://spaces.ac.cn/archives/7388)、[bert-distillation](https://github.com/elephantmipt/bert-distillation)

​		使用WRD进行语义相似度计算。使用压缩代码，将K-BERT压缩成DistilBert，利用DistilBert作为系统encoder。GPT2作为系统decoder的对话模型

​		大致工作流程：输入 -> WRD -> DistilBert -> GPT2

​		前端界面使用的是[BotUI](https://github.com/botui)，后台Flask。

### 项目结构说明：

```
chatbot_XiaoMo

1、主代码
├── brain #参考K-BERT说明
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── CnDbpedia.spo
│   │   ├── HowNet.spo
│   │   └── Medical.spo
│   └── knowgraph.py
├── env # python虚环境
├── models
│   ├──bert_origin # 来源于K-BERT
│   ├──distilbert-chinese #自己提取
│   ├──gpt2_dialogue_model # 来源于GPT2-chitchat
│   └──gpt2_mmi_model # 来源于GPT2-chitchat
├── outputs #存一些输入输出结果
├── uer
│   ├──__init__.py
│   └──optimizers.py #一个调度器
├── chat.py # 运行整个对话系统，整个系统的核心
├── constants.py
├── enc_dec.py # 对话系统模型
├── modeling_distilbert.py # distilBert模型
├── sentences_retrieval.py # WRD算法
├── utils.py # 工具函数
├── README.md

2、训练部分的代码
├── run_kbert_ner.py # 未完成
├── enc_dec_train.ipynb #训练对话系统模型
├── run_distilbert_cls.ipynb
├── run_kbert_cls.ipynb

3、其他代码
├── CDial_chat.ipynb # 单独运行，测试CDial模型
├── data_process_neo4j.py #将CnDbpedia.spo处理成csv数据，导入neo4j4.2.1
├── CnDbpedia.csv
├── CnDbpedia_entity.csv
├── CnDbpedia_relation.csv

4、前后端
├── app.py # flask代码
├── .env
├── .flaskenv
├── templates
└── static
```

1+2=实验分析，1+4=对话系统

GPU环境，显存在11GB以上

CPU环境，内存在8GB以上

### 环境要求：

torch 1.5.1

transformers 3.3.1

Flask

pkuseg

sklearn

numpy

tensorboard

scipy 1.6.1

### 下载说明：

数据集来自于[K-BERT](https://github.com/autoliuweijie/K-BERT)、[CDial-GPT](https://github.com/thu-coai/CDial-GPT)

env按环境要求自己建立就行了,有一些零碎的库按报错信息安装就行

models、brain.kgs自行去[K-BERT](https://github.com/autoliuweijie/K-BERT)、[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)下载相关数据

### 系统运行截图：



