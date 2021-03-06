import pandas as pd
from pandas import Series, DataFrame

## convert spo file to csv file
spo_file_paths = ["D:/AllProject/K-BERT/brain/kgs/CnDbpedia.spo"]
csv_save_path = "D:/AllProject/K-BERT/CnDbpedia.csv"
# csv_list = list()
# for spo_path in spo_file_paths:
#     print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
#     with open(spo_path, 'r', encoding='utf-8') as f:
#         for line in f:
#                 subj, pred, obje = line.strip().split("\t")
#                 csv_list.append([subj, pred, obje])

# with open(csv_save_path, 'w', encoding='utf-8') as f:
#     for i, csv_line in enumerate(csv_list):
#         if i < 20000:
#             f.writelines(','.join(csv_line))
#             f.writelines('\n')

## process csv file to fit neo4j format
entity_list = dict()
csv_entity_path = "D:/AllProject/K-BERT/CnDbpedia_entity.csv"
csv_relation_path = "D:/AllProject/K-BERT/CnDbpedia_relation.csv"

with open(csv_relation_path, 'w', encoding='utf-8') as f1:
    with open(csv_save_path, 'r', encoding='utf-8') as f2:
        i = 0 # entity id
        for line in f2:
            subj, pred, obje = line.strip().split(",")
            if subj not in entity_list.keys():
                entity_list[subj] = i
                i = i + 1
            if obje not in entity_list.keys():
                entity_list[obje] = i
                i = i + 1
            f1.writelines(f'{entity_list[subj]},{pred},{entity_list[obje]}')
            f1.writelines('\n')

with open(csv_entity_path, 'w', encoding='utf-8') as f:
    for k, v in entity_list.items():
        f.writelines(f'{v}, {k}')
        f.writelines('\n')