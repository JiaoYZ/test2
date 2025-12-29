from abc import ABC
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import dgl
from dgl.nn.pytorch import GATConv
import numpy as np


# ----------------- for multiple meta-path fusion ------------------------


# dkt模型走的
class concat_nonLinear(nn.Module, ABC):  # concat + nonLinear
    def __init__(self, args, num_metaPath):
        super(concat_nonLinear, self).__init__()
        #2
        self.num_path = num_metaPath
        # 128*2 -> 128
        self.fc = nn.Linear(args.emb_dim * num_metaPath, args.emb_dim)
        # print(111)
        # self.fc.weight.requires_grad = False

    def forward(self, ques_embeddings):
        # 预训练节点的嵌入  拼接起来
        ques_embeddings = torch.cat(ques_embeddings, dim=-1)
        # 128*2 -> 128
        ques_embedding = F.relu(self.fc(ques_embeddings))
        return ques_embedding
