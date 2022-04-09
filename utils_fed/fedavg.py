import copy
import torch
from torch.nn import functional as F


def fedrated_avg(w, local_test_acc=None):
    # 平均聚合
    if local_test_acc is None:
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
    # 加权聚合
    else:
        if len(w) == len(local_test_acc):
            print("加权聚合")
            w_score = F.softmax(torch.tensor(local_test_acc))
            w_avg = copy.deepcopy(w[0])
            # w_avg = torch.mul(copy.deepcopy(w[0]), w_score[0])
            for k in w_avg.keys():  # w_avg表示每个用户的权重size，依次取出第一维进行权重分配
                w_avg[k] = torch.mul(w_avg[k], w_score[0])
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k] * w_score[i]
            return w_avg


