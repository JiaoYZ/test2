from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ques_emb.SemAttn import *


# ----------------- for single meta-path ------------------------
class PreEmb(nn.Module, ABC):
    #  题目嵌入的融合，不同元路径，每个元路径下都对应同一个题目的不同embedding
    def __init__(self, args, device):
        super(PreEmb, self).__init__()
        # [qkq, qtq]
        self.metaPaths = args.meta_paths
        # 2
        self.num_metaPath = len(self.metaPaths)
        # 为每一个元路径创建一个字典{ 元路径 ； 矩阵（预训练好的） }
        self.mp2quesEmbMat_dict = dict()
        # 选取一个元路径范式
        for mp in self.metaPaths:
            # 此处可以选择window-size是3还是5  默认emb_dim 是 128 加载元路径范式下的节点预训练嵌入
            embPath = "%s/pre_emb/%s/emb/%s_10_80_%d_3.emb.npy" % (args.root,args.data_set, mp, args.emb_dim)
            # values是加载预训练的嵌入  转换为torch tensor  放到device上  字典key是元路径范式名
            self.mp2quesEmbMat_dict[mp] = torch.from_numpy(np.load(embPath)).to(device)

        self.fusion = args.fusion
        assert self.fusion in ['attnVec_dot', 'attnVec_dot_fc', 'attnVec_nonLinear', 'attnVec_nonLinear_fc',
                               'attnVec_topK', 'concat_nonLinear', 'sa_concat_nonLinear', 'attnMat_nonLinear']
        if self.fusion == 'attnVec_dot':  # method 1: train attention vector + dot
            self.semantic_attention = attnVec_dot(args, self.num_metaPath, device)
        # 选择这个融合方式： 将两个元路径，一道题就有两个embedding 拼接起来放入全连接层
        elif self.fusion == 'attnVec_nonLinear':  # method 2: train attention vector + nonLinear
            self.semantic_attention = attnVec_nonLinear(args, self.num_metaPath, device)

        self.whole_ques_embedding = None    
        # self.semantic_attention.weight.requires_grad = False
      
        

    def forward(self, batch_ques):
        # 两个范式下的节点嵌入进行融合
        self.whole_ques_embedding = self.semantic_attention(list(self.mp2quesEmbMat_dict.values()))
        
        batch_ques_embedding = F.embedding(batch_ques, self.whole_ques_embedding)
        
        return batch_ques_embedding
