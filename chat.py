#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import codecs
import collections
from collections import OrderedDict
import copy
from datetime import datetime
from itertools import zip_longest, chain
import json
import logging
from multiprocessing import Process, Pool
import numpy as np
import os
from os.path import join, exists

import random
from sklearn.model_selection import train_test_split
import sys
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (BertConfig, BertTokenizer, 
                          BertModel, BertPreTrainedModel,
                          DistilBertConfig, DistilBertTokenizer,
                          GPT2Config, GPT2LMHeadModel,
                          AdamW, get_linear_schedule_with_warmup)
from typing import Dict, List

from modeling_distilbert import (DistilBertModel,
                                 DistilBertPreTrainedModel)
from enc_dec import EncDecModel
from brain import KnowledgeGraph
from constants import *
from sentences_retrieval import (sentence_retrival_wordlevel,
                                sentence_retrival_sentlevel)
from utils import (set_seed, create_logger, save_model,
                    calculate_loss_and_accuracy)

# arguments setting
def load_hyperparam(args):
    with codecs.open(args.enc_config_path, "r", "utf-8") as f:
        param = json.load(f)
    args.emb_size = param.get("emb_size", 768)
    args.hidden_size = param.get("hidden_size", 768)
    args.kernel_size = param.get("kernel_size", 3)
    args.block_size = param.get("block_size", 2)
    args.feedforward_size = param.get("feedforward_size", None)
    args.heads_num = param.get("heads_num", None)
    args.layers_num = param.get("layers_num", 12)
    args.dropout = param.get("dropout", 0.1)
    
    return args

args = {
    "enc_model_path": "./models/distilbert-chinese/",
    "enc_config_path": "./models/distilbert-chinese/config.json",
    # "enc_model_path": "./models/bert_origin/",
    # "enc_config_path": "./models/bert_origin/config.json",
    "dec_model_path": "./models/gpt2_dialogue_model/",
    "dec_config_path": "./models/gpt2_dialogue_model/config.json",
    # "dec_model_path": "./models/CDial-GPT2_LCCC-base",
    
    "encdec_model_path": "./outputs/encdec_STC_CnDbpedia/model_epoch1",  
    "mmi_model_path": "./models/gpt2_mmi_model",

    "train_path": "/input/datasets_K-BERT/STC-corpus/STC.json",
#     "dev_path":  "/input/datasets_K-BERT/book_review/dev.tsv",
    "test_path":  "/input/datasets_K-BERT/STC-corpus/STC_test.json",

  
    "kg_name": "CnDbpedia",
    "log_path": "./outputs/enc_dec_log.txt",
    "tb_writer_dir": "/output",
    "save_samples_path": "./outputs/sample/", #保存聊天记录的路径

    "batch_size": 5, #生成的回复数量
    "seq_length": 256,
    "learning_rate":2e-5 , # 2e-5, 5e-5
    "warmup": 0.1,
    "dropout": 0.5,
    "epochs_num": 2, # 5, 10, 20
    "log_step": 10, #多少步汇报一次loss
    "max_grad_norm": 1.0, #梯度裁剪
    "gradient_accumulation": 2,  #每n次反向传播/batch，进行一次梯度下降
    "seed": 7,
    "mean_reciprocal_rank": False, # True for DBQA dataset
    "workers_num": 1, # number of process for loading dataset，取决于cpu数量和线程数量
    "no_vm": False, # Disable the visible_matrix
    "temperature": 1, # 对话温度
    "topk": 8,
    "topp": 0, # 核方法
    "repetition_penalty": 1.0, #惩罚重复性回复
    "max_history_len": 5, # 对话历史保存长度
    "all_response": True, #指定该参数，可以查看生成的所有候选的reponse，及其loss
    "max_len": 25 #每个回复的最大长度,超过指定长度则进行截断
}

class Args(dict):  #字典转对象，递归版,既可以作为对象、也可以作为属性
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
args = Args(args)
args = load_hyperparam(args) # Load the hyperparameters from the config file.

# basic setting
logger = create_logger(args)
set_seed(args.seed)

# Build knowledge graph.
if args.kg_name == 'none':
    spo_files = []
else:
    spo_files = [args.kg_name]
kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


# In[ ]:


# 当用户使用GPU,并且GPU可用时
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info('using device:{}'.format(device))
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device

## encoder distilbert origin
enc_model_config = DistilBertConfig.from_pretrained(args.enc_model_path)
enc_model_token = BertTokenizer.from_pretrained(args.enc_model_path)
enc_model = DistilBertModel(config=enc_model_config)
enc_model.config.max_position_embeddings = args.seq_length #句子最大长度256
# enc_model_config = BertConfig.from_pretrained(args.enc_model_path)
# enc_model_token = BertTokenizer.from_pretrained(args.enc_model_path)
# enc_model = BertModel(config=enc_model_config)
# enc_model.config.max_position_embeddings = args.seq_length #句子最大长度256

## decoder model
dec_model_token = BertTokenizer.from_pretrained(args.dec_model_path)
dec_model = GPT2LMHeadModel.from_pretrained(args.dec_model_path)

# 对话model
encdec_model = EncDecModel(config=enc_model_config, args=args, enc_model=enc_model, dec_model=dec_model)
# encdec_model.load_state_dict(torch.load("/data/K-BERT/outputs/encdec_STC_CnDbpedia/model_epoch1/pytorch_model.bin"))
encdec_model.to(device)
encdec_model.eval()

# 互信息mmi model
mmi_model = GPT2LMHeadModel.from_pretrained(args.mmi_model_path)
mmi_model_token = BertTokenizer.from_pretrained(args.mmi_model_path)
mmi_model.to(device)
mmi_model.eval()


# In[ ]:


if args.save_samples_path:
    if not os.path.exists(args.save_samples_path):
        os.makedirs(args.save_samples_path)
    samples_file_cls = open(args.save_samples_path + '/mmi_samples_cls.txt', 'a', encoding='utf8')
    samples_file_rawtext = open(args.save_samples_path + '/mmi_samples_rawtext.txt', 'a', encoding='utf8')
    samples_file_cls.write("聊天记录{}:\n".format(datetime.now()))
    samples_file_rawtext.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储

history = [] # list of 4-tuple, (plain text, plain text id, list of embeddings, [CLS] hidden states)
resp_history = []
print('开始和chatbot聊天，输入CTRL + C以退出')


# In[ ]:


### decoder with [CLS] token only
def chat_cls(text :str)->dict:
    global history, resp_history
    result = {
        'scores_w': None,
        'scores_s': None,
        'scores' :None,
        'history_input': None,
        'kg_input': None,
        'candidate_response': [],
        'response': None
    }

    try:
        start_time = datetime.now()
        if args.save_samples_path:
            samples_file_cls.write("user:{}\n".format(text))

        ## 提取历史对话
#         text_ids = enc_model_token.encode(text)
        text_ids = enc_model_token.tokenize(text) # str2list, 自带数据清洗和特殊标志识别
        text_ids = enc_model_token.convert_tokens_to_ids(text_ids) # list of str -> list of id
        input_ids = [enc_model_token.cls_token_id] + text_ids
        embeddings = encdec_model.enc.embeddings.word_embeddings(torch.tensor(text_ids,device=device))
        cls_hidden_state = encdec_model.enc(input_ids=torch.tensor([input_ids],device=device))[0][0][0]
            # [0][0][0]代表tuple中的hidden state， batch中的第一句， hidden state中的第一个单词[CLS]
        history_data = (text, text_ids, embeddings.detach(), cls_hidden_state.detach())
            
        if history:
            ''' 对话历史抽取 '''
            scores_w = sentence_retrival_wordlevel(embeddings.detach().cpu(), [e[2].cpu() for e in history])
            scores_s = sentence_retrival_sentlevel(cls_hidden_state.detach().cpu(), [e[3].cpu() for e in history])
            print(scores_w, scores_s)
            scores = 0.5 * scores_w + 0.5 * scores_s
            history_text = history[np.argmax(scores).item()][0]
            text = text + SEP_TOKEN + history_text + SEP_TOKEN
            print('历史抽取后: ', CLS_TOKEN + text)
            result['scores_w'] = str(scores_w)
            result['scores_s'] = str(scores_s)
            result['scores'] = str(scores)
            result['history_input'] = '历史抽取后: ' + CLS_TOKEN + text

        history.append(history_data)
        history = history[:args.max_history_len]

        history_time = datetime.now()
        ## 嵌入知识图谱
        text = CLS_TOKEN + text
        tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length) # modified
        tokens = tokens[0]
        pos = pos[0]
        vm = vm[0]
        token_ids = enc_model_token.convert_tokens_to_ids(tokens)
        seg = []
        seg_tag = 0
        for t in tokens:
            seg.append(seg_tag)
            if t == SEP_TOKEN:
                seg_tag = 1 # modified, because token_type_id can only be 0 or 1, padding token_type_id is not important

        knowledge_time = datetime.now()
        print("知识嵌入后：", "".join(t for t in tokens if t != '[PAD]'))
        result['kg_input'] = "知识嵌入后：" + "".join(t for t in tokens if t != '[PAD]')
        token_ids = torch.LongTensor([token_ids]).to(device)
        vm = torch.LongTensor([vm]).to(device)
        pos = torch.LongTensor([pos]).to(device)
        seg = torch.LongTensor([seg]).to(device)

        cls_hidden_state = encdec_model.enc(
            input_ids=token_ids, 
            attention_mask=vm,
            position_ids=pos,
            token_type_ids=seg
        )[0][0][0]

        cls_hidden_state = cls_hidden_state.clone().detach().unsqueeze(0).cpu().numpy() # (768) -> (1, 768)

        encoder_time = datetime.now()
        print("encoder time cosume: ", encoder_time - knowledge_time)
        # 用于批量生成response，维度为(batch_size,token_len)
        cls_hidden_state = [copy.deepcopy(cls_hidden_state) for _ in range(args.batch_size)]

        curr_input_tensors = torch.FloatTensor(cls_hidden_state).to(device)
        
        generated = []  # 二维数组，维度为(生成的response的最大长度，batch_size)，generated[i,j]表示第j个response的第i个token的id
        finish_set = set()  # 标记是否所有response均已生成结束，若第i个response生成结束，即生成了sep_token_id，则将i放入finish_set
        # 最多生成max_len个token
        for _ in range(args.max_len):
            outputs = encdec_model.dec(inputs_embeds=curr_input_tensors)
            next_token_logits = outputs[0][:, -1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            for next_token_logit in next_token_logits:
                next_token_logit[dec_model_token.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            # 判断是否有response生成了[SEP],将已生成了[SEP]的resposne进行标记
            for index, token_id in enumerate(next_token[:, 0]):
                if token_id == dec_model_token.sep_token_id:
                    finish_set.add(index)
            # 检验是否所有的response均已生成[SEP]
            finish_flag = True  # 是否所有的response均已生成[SEP]的token
            for index in range(args.batch_size):
                if index not in finish_set:  # response批量生成未完成
                    finish_flag = False
                    break
            if finish_flag:
                break
            generated.append([token.item() for token in next_token[:, 0]])
            # 将新生成的token与原来的token进行拼接
            next_token = encdec_model.dec.transformer.wte(next_token)
            
            curr_input_tensors = torch.cat((curr_input_tensors, next_token), dim=1)

        candidate_responses = []  # 生成的所有候选response
        for batch_index in range(args.batch_size):
            response = []
            for token_index in range(len(generated)):
                if generated[token_index][batch_index] != dec_model_token.sep_token_id:
                    response.append(generated[token_index][batch_index])
                else:
                    break
            candidate_responses.append(response)
        
        decoder_time = datetime.now()
#         print("decoder time cosume: ", decoder_time - encoder_time)
        # mmi模型的输入
        if args.all_response:
            print("candidate response:")
        samples_file_cls.write("candidate response:\n")
        min_loss = float('Inf')
        best_response = ""
        for response in candidate_responses:
            mmi_input_id = [mmi_model_token.cls_token_id]  # 每个input以[CLS]为开头
            mmi_input_id.extend(response)
            mmi_input_id.append(mmi_model_token.sep_token_id)
            for history_utr in reversed(history[-args.max_history_len:]):
                mmi_input_id.extend(history_utr[1])
                mmi_input_id.append(mmi_model_token.sep_token_id)
            mmi_input_tensor = torch.tensor(mmi_input_id).long().to(device)
            out = mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)
            loss = out[0].item()
            if args.all_response:
                text = dec_model_token.convert_ids_to_tokens(response)
                # print("{} loss:{}".format("".join(text), loss))
                result['candidate_response'].append("{} loss:{}".format("".join(text), loss))
            samples_file_cls.write("{} loss:{}\n".format("".join(text), loss))
            if loss < min_loss:
                best_response = response
                min_loss = loss
        resp_history.append(best_response)
        text = dec_model_token.convert_ids_to_tokens(best_response)
        print("chatbot:" + "".join(text))
        result['response'] = "".join(text)
        if args.save_samples_path:
            samples_file_cls.write("chatbot:{}\n".format("".join(text)))

        mmi_time = datetime.now()
    except KeyboardInterrupt:
        if args.save_samples_path:
            samples_file_cls.close()
        return "KeyboardInterrupt"

    return result


### decoder with raw text
def chat_rawtext(text: str)->dict: 
    global history, resp_history
    result = {
        'scores_w': None,
        'scores_s': None,
        'scores' :None,
        'history_input': None,
        'kg_input': None,
        'candidate_response': [],
        'response': None
    }

    try:
        if args.save_samples_path:
            samples_file_rawtext.write("user:{}\n".format(text))

        ## 提取历史对话
#         text_ids = enc_model_token.encode(text)
        text_ids = enc_model_token.tokenize(text) # str2list, 自带数据清洗和特殊标志识别
        text_ids = enc_model_token.convert_tokens_to_ids(text_ids) # list of str -> list of id
        input_ids = [enc_model_token.cls_token_id] + text_ids
        embeddings = encdec_model.enc.embeddings.word_embeddings(torch.tensor(text_ids,device=device))
        cls_hidden_state = encdec_model.enc(input_ids=torch.tensor([input_ids],device=device))[0][0][0]
            # [0][0][0]代表tuple中的hidden state， batch中的第一句， hidden state中的第一个单词[CLS]
        history_data = (text, text_ids, embeddings.detach(), cls_hidden_state.detach())
            
        if history:
            ''' 对话历史抽取 '''
            scores_w = sentence_retrival_wordlevel(embeddings.detach().cpu(), [e[2].cpu() for e in history])
            scores_s = sentence_retrival_sentlevel(cls_hidden_state.detach().cpu(), [e[3].cpu() for e in history])
            print(scores_w, scores_s)
            scores = 0.5 * scores_w + 0.5 * scores_s
            history_text = history[np.argmax(scores).item()][0]
            text = text + SEP_TOKEN + history_text + SEP_TOKEN
            print('历史抽取后: ', CLS_TOKEN + text)
            result['scores_w'] = str(scores_w)
            result['scores_s'] = str(scores_s)
            result['scores'] = str(scores)
            result['history_input'] = '历史抽取后: ' + CLS_TOKEN + text

        history.append(history_data)
        history = history[:args.max_history_len]

        ## 嵌入知识图谱
        text = CLS_TOKEN + text
        tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length) # modified
        tokens = tokens[0]
        pos = pos[0]
        vm = vm[0]
        token_ids = enc_model_token.convert_tokens_to_ids(tokens)
        seg = []
        seg_tag = 0
        for t in tokens:
            seg.append(seg_tag)
            if t == SEP_TOKEN:
                seg_tag = 1 # modified, because token_type_id can only be 0 or 1, padding token_type_id is not important

        print("知识嵌入后：", "".join(t for t in tokens if t != '[PAD]'))
        result['kg_input'] = "知识嵌入后：" + "".join(t for t in tokens if t != '[PAD]')
        token_ids = torch.LongTensor([token_ids]).to(device)
        vm = torch.LongTensor([vm]).to(device)
        pos = torch.LongTensor([pos]).to(device)
        seg = torch.LongTensor([seg]).to(device)
        
        cls_hidden_state = encdec_model.enc(
            input_ids=token_ids, 
            attention_mask=vm,
            position_ids=pos,
            token_type_ids=seg
        )[0][0][0]

        cls_hidden_state = cls_hidden_state.clone().detach().unsqueeze(0).cpu().numpy() # (768) -> (1, 768)
        
        ## decoder端输入加上text
        text_ids = dec_model_token.tokenize(text)
        text_ids = dec_model_token.convert_tokens_to_ids(text_ids)
        curr_text_tensors = encdec_model.dec.transformer.wte(torch.tensor(text_ids,device=device)) # (seq_len, 768), 无[CLS]
        curr_cls_tensors = torch.FloatTensor(cls_hidden_state).to(device) # (1, 768)
        curr_input_tensors = torch.cat((curr_cls_tensors, curr_text_tensors),dim=0) # (seq_len, 768)
        
        # 用于批量生成response，维度为(batch_size,token_len)
        curr_input_tensors = [copy.deepcopy(cls_hidden_state) for _ in range(args.batch_size)]
        curr_input_tensors = torch.FloatTensor(curr_input_tensors).to(device) # (5, seq_len, 768)
        
        generated = []  # 二维数组，维度为(生成的response的最大长度，batch_size)，generated[i,j]表示第j个response的第i个token的id
        finish_set = set()  # 标记是否所有response均已生成结束，若第i个response生成结束，即生成了sep_token_id，则将i放入finish_set
        # 最多生成max_len个token
        for _ in range(args.max_len):
            outputs = encdec_model.dec(inputs_embeds=curr_input_tensors)
            next_token_logits = outputs[0][:, -1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            for next_token_logit in next_token_logits:
                next_token_logit[dec_model_token.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            # 判断是否有response生成了[SEP],将已生成了[SEP]的resposne进行标记
            for index, token_id in enumerate(next_token[:, 0]):
                if token_id == dec_model_token.sep_token_id:
                    finish_set.add(index)
            # 检验是否所有的response均已生成[SEP]
            finish_flag = True  # 是否所有的response均已生成[SEP]的token
            for index in range(args.batch_size):
                if index not in finish_set:  # response批量生成未完成
                    finish_flag = False
                    break
            if finish_flag:
                break
            generated.append([token.item() for token in next_token[:, 0]])
            # 将新生成的token与原来的token进行拼接
            next_token = encdec_model.dec.transformer.wte(next_token)
            
            curr_input_tensors = torch.cat((curr_input_tensors, next_token), dim=1)

        candidate_responses = []  # 生成的所有候选response
        for batch_index in range(args.batch_size):
            response = []
            for token_index in range(len(generated)):
                if generated[token_index][batch_index] != dec_model_token.sep_token_id:
                    response.append(generated[token_index][batch_index])
                else:
                    break
            candidate_responses.append(response)
        
        # mmi模型的输入
        if args.all_response:
            print("candidate response:")
        samples_file_rawtext.write("candidate response:\n")
        min_loss = float('Inf')
        best_response = ""
        for response in candidate_responses:
            mmi_input_id = [mmi_model_token.cls_token_id]  # 每个input以[CLS]为开头
            mmi_input_id.extend(response)
            mmi_input_id.append(mmi_model_token.sep_token_id)
            for history_utr in reversed(history[-args.max_history_len:]):
                mmi_input_id.extend(history_utr[1])
                mmi_input_id.append(mmi_model_token.sep_token_id)
            mmi_input_tensor = torch.tensor(mmi_input_id).long().to(device)
            out = mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)
            loss = out[0].item()
            if args.all_response:
                text = dec_model_token.convert_ids_to_tokens(response)
                print("{} loss:{}".format("".join(text), loss))
                result['candidate_response'].append("{} loss:{}".format("".join(text), loss))
            samples_file_rawtext.write("{} loss:{}\n".format("".join(text), loss))
            if loss < min_loss:
                best_response = response
                min_loss = loss
        resp_history.append(best_response)
        text = dec_model_token.convert_ids_to_tokens(best_response)
        print("chatbot:" + "".join(text))
        result['response'] = "".join(text)
        if args.save_samples_path:
            samples_file_rawtext.write("chatbot:{}\n".format("".join(text)))
    except KeyboardInterrupt:
        if args.save_samples_path:
            samples_file_rawtext.close()
        return "KeyboardInterrupt"

    return result


# In[ ]:


# text_list = [
#     '你好',
#     '哦哦，好的。你读过什么书吗，比如傅雷家书之类的书籍。你还有别的兴趣爱好吗。比如打球、射箭、跑步之类的活动',
#     '哈哈哈',
#     '是这样吗。你知道北京天安门怎么走吗，我一直很想去那旅游，但是一直没有时间，省察自己的人生，我们当然可以像大树下酣睡初醒的庄子那样提问：到底是我变成了蝴蝶，还是蝴蝶变成了我？但更好的方式，我想是通过他人的目光来审视自我。，真的有点苦恼哎，你有啥建议吗',
#     '好的，明白了',
#     '生活在现实世界里，我们少不得彼此依靠、相互影响。因此，我们不妨"我看人看我"，好好地想一想他人如何看待我们自己的生活、我们又会如何理解他人的生活。在相互比较和沟通中，我们或许会发现，这个世界其实并不像萨特所说"他人即地狱"。',
#     '测试测试',
#     '我们每个人都有独具特点的一生，每个人都有别人无法复制的一生，同时我们每个人也有与他人息息相关的一生。所谓"以人为镜可以知得失"',
#     '你再说说，还没懂',
#     '省察自己的人生，我们当然可以像大树下酣睡初醒的庄子那样提问：到底是我变成了蝴蝶，还是蝴蝶变成了我？但更好的方式，我想是通过他人的目光来审视自我。'
# ]


for text in text_list:
    print(chat_cls(text)['response'])
    print("chat cls------------------------------------------------------------------------------")
    print(chat_rawtext(text)['response'])
    print("chat rawtext--------------------------------------------------------------------------")
# while True:
#     text = input('请输入对话: ')
#     print(chat_cls(text)['response'])
#     print(chat_rawtext(text)['response'])

