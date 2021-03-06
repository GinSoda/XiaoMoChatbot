'''
referenced from https://kexue.fm/archives/7388
'''
import torch
import numpy as np
from scipy.optimize import linprog
import torch.nn as nn
cosSimi = nn.CosineSimilarity(dim=0, eps=1e-6)

def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun

def word_rotator_distance(x, y):
    """WRD（Word Rotator's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    # print(f"x:{x}, y:{y}") ##
    # print(f"p:{p}, q:{q}, D:{D}") ##
    return wasserstein_distance(p, q, D)


def word_rotator_similarity(x, y):
    """1 - WRD
    x.shape=[m,d], y.shape=[n,d]
    """
    return 1 - word_rotator_distance(x, y)

def sentence_retrival_wordlevel(now, history: list):
    '''
    now: embeddings
    history: list of embeddings
    '''
    scores = []
    # now = now[1:] #去掉[CLS] token
    for idx in range(len(history)):
        # history[idx] = history[idx][1:] #去掉[CLS] token
        scores.append(word_rotator_similarity(now, history[idx]))
    return np.array(scores)

def sentence_retrival_sentlevel(now, history: list):
    '''
    now: [CLS] hidden state
    history: list of [CLS] hidden state
    '''

    scores = []
    for idx in range(len(history)):
        scores.append(cosSimi(now, history[idx]))
    return np.array(scores) 
