#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join, exists
import collections
import codecs
import sys
import json
import random
from typing import Dict, List
from multiprocessing import Process, Pool
from collections import OrderedDict
import logging
from datetime import datetime
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (BertConfig, BertTokenizer, 
                          BertModel, BertPreTrainedModel,
                          DistilBertConfig, DistilBertTokenizer,
                          GPT2Config, GPT2LMHeadModel,
                          AdamW, get_linear_schedule_with_warmup)

from modeling_distilbert import (DistilBertModel, 
                                 DistilBertPreTrainedModel)

from uer.optimizers import BertAdam
from brain import KnowledgeGraph
from utils import (set_seed, create_logger, save_model, 
                    calculate_loss_and_accuracy)
from constants import * 

# build model
## whole model
class EncDecModel(transformers.PreTrainedModel):
    def __init__(self, config, args, enc_model, dec_model):
        super().__init__(config)
        self.enc = enc_model
        self.dec = dec_model
    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        position_ids=None, 
        token_type_ids=None, 
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        label_ids = None
    ):
        
        enc_output = self.enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids, 
            token_type_ids=token_type_ids, 
            head_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        enc_hidden_states = enc_output[0]
        label_embeds = self.dec.transformer.wte(label_ids) # id2embedding
        label_embeds[:,0] = enc_hidden_states[:,0] # 只替换[CLS]
        outputs = self.dec(inputs_embeds=label_embeds)

        return outputs


