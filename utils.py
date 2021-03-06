import sys
import os
import codecs
import logging
import torch
from torch.nn import CrossEntropyLoss
import json
import random
import math
import numpy as np
from constants import * 

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def load_hyperparam(args):
    with codecs.open(args.config_path, "r", "utf-8") as f:
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

def save_model(model, model_path):
    # We dont't need prefix "module".
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    # We dont't need task layer either.
    elif hasattr(model, "origin"):
        torch.save(model.origin.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def word2sub(word_ids, vocab, sub_vocab, subword_type):
    '''
    word_ids: batch_size, seq_length
    '''
    batch_size, seq_length = word_ids.size()
    device = word_ids.device
    word_ids = word_ids.contiguous().view(-1).tolist()
    words = [vocab.i2w[i] for i in word_ids]
    max_length = max([len(w) for w in words])
    sub_ids = torch.zeros((len(words), max_length), dtype=torch.long).to(device)
    for i in range(len(words)):
        for j, c in enumerate(words[i]):
            sub_ids[i, j] = sub_vocab.w2i.get(c, UNK_ID)
    return sub_ids

def calculate_loss_and_accuracy(outputs, labels, device):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(PAD_ID)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy