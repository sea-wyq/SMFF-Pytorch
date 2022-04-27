"""
@author:32369
@file:metric.py
@time:2022/03/19
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def dcg_score(y_true, y_score, k=5):
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=5):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def validateStep(labels, preds, indexs):
    auc, mrr, ndcg5, ndcg10 = 0, 0, 0, 0
    s = 0
    auc = roc_auc_score(labels, preds)
    for i in indexs[1:]:
        label, pred = labels[s:i], preds[s:i]
        mrr += mrr_score(label, pred)
        ndcg5 += ndcg_score(label, pred, k=5)
        ndcg10 += ndcg_score(label, pred, k=10)
        s = i
    return auc, mrr / len(indexs), ndcg5 / len(indexs), ndcg10 / len(indexs)
