# coding: utf-8
"""
KnowledgeGraph
"""
import os
import brain.config as config
# import config
import pkuseg
import numpy as np


class KnowledgeGraph(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, spo_files, predicate=False):
        self.predicate = predicate
        self.spo_file_paths = [config.KGS.get(f, f) for f in spo_files]
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)
        self.special_tags = set(config.NEVER_SPLIT_TAG) # 特殊标记

    def _create_lookup_table(self):
        lookup_table = {}
        for spo_path in self.spo_file_paths: #依次打开三个spo文件
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = pred + obje
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value]) # lookup_table是一个字典，keys是subj，values是一个set,set中顺序不固定
        return lookup_table

    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        split_sent_batch = [self.tokenizer.cut(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []
        for split_sent in split_sent_batch:

            # create tree
            sent_tree = [] # list of 2-tuple
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1 #相对位置
            abs_idx = -1 #绝对位置
            abs_idx_src = []
            for token in split_sent: # 对句子中的每个单词

                entities = list(self.lookup_table.get(token, []))[:max_entities]
                sent_tree.append((token, entities))

                if token in self.special_tags:
                    token_pos_idx = [pos_idx+1]
                    token_abs_idx = [abs_idx+1]
                else:
                    token_pos_idx = [pos_idx+i for i in range(1, len(token)+1)]
                    token_abs_idx = [abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1] #更新绝对位置

                entities_pos_idx = []
                entities_abs_idx = []
                for ent in entities: # 一个head实体对应多个tail实体
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent)+1)]
                    entities_pos_idx.append(ent_pos_idx) #一个head实体对应所有tail实体的相对位置
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent)+1)]
                    abs_idx = ent_abs_idx[-1] #更新绝对位置
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                word = sent_tree[i][0]
                if word in self.special_tags:
                    know_sent += [word] # 特殊字符作为单个token
                    seg += [0]
                else:
                    add_word = list(word) # 分词结果作为多个token
                    know_sent += add_word
                    seg += [0] * len(add_word)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    add_word = list(sent_tree[i][1][j])
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += list(pos_idx_tree[i][1][j])

            token_num = len(know_sent)

            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]
            
            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)
        
        # position_batch代表相对位置
        # seg_batch种0代表原句子，1代表插入的知识
        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch

def test(): # test
    spo_files = ["HowNet","Medical"]
    kg = kg = KnowledgeGraph(spo_files=spo_files, predicate=True)
    sent_batch = [
        "1796年至1800年，正是贝多芬的耳疾开始越来越深地影响着他的时期，贝多芬的耳朵日夜作响。",
        "贝多芬的作品总是根植于所生活的现实之中，贝多芬在现实中承受着一切痛苦，享受着每一份欢乐，并将它们表现在他的作品之中。",
        "尽管贝多芬在恋爱上不断地钟情，不断地梦想着幸福，最后幸福却总是幻灭，使贝多芬陷入痛苦的煎熬之中，贝多芬却仍旧一次又一次地坠入爱河之中",
        "并将月光奏鸣曲献给当时的恋爱对象琪里爱太·吉却娣，然而，这位漂亮轻佻的贵族小姐却最终无情地抛弃了贝多芬。",
        "正是在这种复杂的痛苦体验和激烈的内心矛盾冲突中，在对爱情的期待与失望之中，才诞生了这部伟大的作品。",
    ]
    # split_sent_batch = [kg.tokenizer.cut(sent) for sent in sent_batch]
    # [print(sent) for sent in split_sent_batch]

    know_sent_batch, position_batch, visible_matrix_batch, seg_batch = kg.add_knowledge_with_vm(sent_batch)
    print("------------------------------------------------------------------")
    [print(i) for i in position_batch]
    print("------------------------------------------------------------------")
    [print(i.shape) for i in visible_matrix_batch]
    print("------------------------------------------------------------------")
    [print(i) for i in seg_batch]

if __name__ == "__main__":
    test()