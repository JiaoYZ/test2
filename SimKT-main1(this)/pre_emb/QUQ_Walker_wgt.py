import numpy as np
import pandas as pd
import time
import os
import torch
# from pre_emb.Wgt_Walker import WeightedWalker
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x):
    return torch.tensor(x).to(device)


class WeightedWalker:
    # 根据边权重，动态地确定节点的采样概率
    # 每调用一次get_next，完成从A类节点（头节点）到B类节点（尾节点）的一步游走，v_num是A类节点的总数，edges是A类节点指向B类节点的边的列表
    def __init__(self, v_num, edges):
        # 记录节点总数
        self.v_num = v_num
        # 为每个节点索引创建一个空列表，记录节点邻居索引
        ngbrs = [[] for i in range(v_num)]
        # 为每个节点索引创建一个空列表，记录节点邻居权重
        weights = [[] for i in range(v_num)]
        for u, v, w in edges:
            # str to int
            u = int(u)
            v = int(v)
            ngbrs[u].append(v)
            weights[u].append(w)

        # 将列表转换成tensor张量，将所有列表填充成等长长度并转化为tensor张量 node_num * ngbrs
        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True)
        # 同上 node_num * weights
        self.weights = pad_sequence([to_tensor(s) for s in weights], batch_first=True)
        # 获取每个节点的邻居数量，用于后续计算概率，形状为 node_num * 1
        self.ngbr_nums = to_tensor([len(ngbr) for ngbr in ngbrs]).unsqueeze(1)
        # 获取所有节点的最大邻居数，方便后续计算概率
        self.max_ngbr_num = torch.max(self.ngbr_nums).item()

        # stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ability.csv"))  # 考虑离散能力值
        # 读取学生能力值，考虑连续能力值
        stu_ability = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_abi_w.csv"))  # 考虑连续能力值
        # 获取学生能力值列
        ability = stu_ability.iloc[:, 1]
        # 计算能力值的平均值
        abi_average = np.average(ability)
        # 将能力值转换为tensor张量，并重塑为列向量
        ability = torch.tensor(ability).reshape(-1, 1)
        # 公式(9)第一个
        self.node_attr_mat = torch.sigmoid(-torch.abs(ability - abi_average)).to(device)  # 正向的
        # self.node_attr_mat = torch.sigmoid(torch.abs(ability - abi_average)).to(device)  # 反向验证
        # 读取问题的区分度
        ques_disc = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_discvalue.csv")) # 考虑题目区分度
        disc = ques_disc.iloc[:,1]
        disc = torch.tensor(disc).reshape(-1, 1)
        self.node_attr_mat1 = torch.sigmoid(disc).to(device)    # 正向的
        # self.node_attr_mat1 = torch.sigmoid(-disc).to(device) # 反例证明
        # 题目区分度差距为1
        self.D = 1
        self.the_nodes = None
    # weights 到邻居节点的权重，nbgrs邻居节点信息，num当前节点实际的邻居数量，the_weights当前节点的能力值或区分度
    def get_prob(self, weights, ngbrs, nums, the_weights,flag):
        if flag == 0:
            # Q-U,考虑学生作答结果一致的情况
            if the_weights is not None:
                # 计算到其他邻居节点的转移概率权重1 公式(8)
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights-the_weights), to_tensor([self.D]))
                # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # prob *= prob2
            else:
                # 如果是起始节点首跳到第一个s节点
                prob = torch.ones(weights.size()).to(device)  # 第一跳，赋予相同的值，进行均匀随机采样
                # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # prob *= prob2
        else:
            # U-Q 考虑题目区分度 也得考虑学生的能力值差异
            if the_weights is not None:
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights - the_weights), to_tensor([self.D]))
                # 考虑题目区分度
                prob2 = F.embedding(ngbrs, self.node_attr_mat1).squeeze(2)
                prob *= prob2

            else:
                # 第一次u-q
                prob = torch.ones(weights.size()).to(device)  # 第一跳，赋予相同的值，进行均匀随机采样
                # 考虑题目区分度
                prob2 = F.embedding(ngbrs, self.node_attr_mat1).squeeze(2)
                prob *= prob2

        # 防止回溯，即防止重复采样同一节点
        if self.the_nodes is not None:
            # 找到上个采样节点的索引
            mask1 = (ngbrs == self.the_nodes)
            # 均值采样的一种方式
            mask2 = torch.rand(nums.size()).to(device) > torch.true_divide(to_tensor([1]), nums)
            prob = prob.masked_fill(mask1 & mask2, 1e-32)  # 赋予一个接近于0的非零值，避免特殊情况下计算prob时分母为零

        # 令填充的部分采样概率为0
        x = torch.unsqueeze(torch.arange(0, self.max_ngbr_num).to(device), 0)
        mask = x >= nums
        prob = prob.masked_fill(mask, 0)

        prob = torch.true_divide(prob, torch.sum(prob, dim=1, keepdim=True))
        return prob  # [v_num * walk_num, max_ngbr_num]



    def get_next(self, v, the_weights,flag):
        # 得到当前节点到邻居节点的权重
        expand_pad_weights = self.weights[v]
        # 得到当前节点的邻居数量
        expand_ngbr_nums = self.ngbr_nums[v]
        # 得到当前节点的邻居节点信息
        expand_pad_ngbrs = self.ngbrs[v]
        # 得到当前节点到邻居节点的转移概率
        expand_pad_prob = self.get_prob(expand_pad_weights, expand_pad_ngbrs, expand_ngbr_nums, the_weights,flag)
        # 按概率采样下一个节点
        next_index = torch.multinomial(expand_pad_prob, num_samples=1)  # 按概率采样
        # [v_num * walk_num, 1]
        self.the_nodes = torch.gather(expand_pad_ngbrs, 1, next_index)
        # [v_num * walk_num]
        next_v = self.the_nodes.squeeze()
        # torch.gather input dim query_index  在input内容的dim维上 拿出query_index对应的元素
        # [v_num * walk_num, 1]
        the_weights = torch.gather(expand_pad_weights, 1, next_index)
        # 得到下一个采样的节点和到达该节点的权重
        return next_v, the_weights


class QUQ_Walker:
    def __init__(self):
        # self.D = 1  # D notates the biggest distance between answer
        self.qu_edge_list = []
        self.uq_edge_list = []
        self.num_stu = None
        self.num_ques = None

        self.read_data()

    def read_data(self):
        # 读取学生-问题-作答结果图
        stu_ques_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "stu_ques.csv"))
        # 获取学生数
        self.num_stu = len(set(stu_ques_df['stu']))
        # 获取问题数
        self.num_ques = len(set(stu_ques_df['ques']))
        print("num of stu: %d" % self.num_stu)
        print("num of ques: %d" % self.num_ques)

        for index, row in stu_ques_df.iterrows():
            stuID, quesID = int(row['stu']), int(row['ques'])
            answerState = row['correct']
            # 记录q-u和u-q边，每个列表中的元素都是一个三元组
            self.qu_edge_list.append((quesID, stuID, answerState))
            self.uq_edge_list.append((stuID, quesID, answerState))
    # 默认的游走次数和路径长度 10 80
    def create_paths(self, walk_num=10, walk_len=80):
        # 构建QU_Walker和UQ_walker 输入参数为节点数和边列表
        QU_Walker = WeightedWalker(v_num=self.num_ques, edges=self.qu_edge_list)
        UQ_Walker = WeightedWalker(v_num=self.num_stu, edges=self.uq_edge_list)
        # 生成每个节点的索引序列，然后复制walk_num次，在dim=1上，这样就形成了walk_num*num_ques，然后再转置，展平
        # [0, 一共walk_num个0]
        next_q = torch.arange(self.num_ques).to(device).repeat(walk_num, 1).T.flatten()
        # 将起始游走节点放入路径列表中
        paths = [next_q]
        # 放回溯的节点信息，初始时为问题节点序列
        UQ_Walker.the_nodes = next_q.unsqueeze(1)
        the_weights = None
        # 随机游走得到walk_len长度的问题节点序列
        for i in range(1, walk_len):
            print("%dth hop" % i)
            next_u, the_weights = QU_Walker.get_next(next_q, the_weights,0)
            next_q, the_weights = UQ_Walker.get_next(next_u, the_weights,1)
            paths.append(next_q)

        paths = [path.unsqueeze(-1) for path in paths]
        paths = torch.cat(paths, dim=-1)
        paths = paths.view(-1, walk_len)
        return paths


if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    for data_set in ["EdNet"]:
        save_folder ="../pre_emb/%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        D_dict = {'qu': 1}
        # decay_rate = 0.5
        walker = QUQ_Walker()
        for num_walks in [10]:
            for walk_length in [80]:
                t = time.time()
                save_file = "walks_quq_wgt_pos6.2_%d_%d.txt" % (num_walks, walk_length)
                with open(os.path.join(save_folder, save_file), 'w') as f:
                    paths1 = walker.create_paths(num_walks, walk_length)
                    for path1 in paths1.cpu().detach().tolist():
                        f.write(','.join([str(e) for e in path1]) + '\n')
                print("time consuming: %d seconds" % (time.time() - t))
