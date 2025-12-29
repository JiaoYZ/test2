import numpy as np
import pandas as pd
import time
import os
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from pre_emb.Wgt_Walker import WeightedWalker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def to_tensor(x):
    return torch.tensor(x).to(device)


class WeightedWalker:
    # 根据边权重，动态地确定节点的采样概率
    # 每调用一次get_next，完成从A类节点（头节点）到B类节点（尾节点）的一步游走，v_num是A类节点的总数，edges是A类节点指向B类节点的边的列表
    def __init__(self, v_num, edges):
        self.v_num = v_num
        # 创建v_num个空列表，每个列表用于存储与节点v相关的邻居节点和权重
        ngbrs = [[] for i in range(v_num)]
        weights = [[] for i in range(v_num)] # 问题难度v
        for u, v, w in edges:
            # 将quesid，skillid抓换成int类型
            u = int(u)
            v = int(v)
            # 添加u节点的邻居列表，以及对应的权重到ngbrs和weights中
            ngbrs[u].append(v)
            weights[u].append(w)
        # 将长度不等的邻居子列表和权重子列表转换为长度相等的tensor张量，不等长的填充
        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True)  # 填充后里邻居
        # 同上
        self.weights = pad_sequence([to_tensor(s) for s in weights], batch_first=True) # 填充后问题难度
        # 统计每个节点的邻居数，并将每个节点的邻居数转换为tensor张量 unsqueeze(1)是将每个节点的邻居数转换为列向量
        self.ngbr_nums = to_tensor([len(ngbr) for ngbr in ngbrs]).unsqueeze(1) # 邻居个数
        # 最大邻居数
        self.max_ngbr_num = torch.max(self.ngbr_nums).item() # 最大邻居数
        # 读取skill_var.csv文件，该文件包含了每个技能的细粒度
        # skill_var = pd.read_csv(os.path.join(data_path, data_set, "graph", "skill_var.csv"))
        # 只读取技能技能的细粒度 并将每个技能细粒度转换为列向量
        # var = skill_var.iloc[:, 1]
        # var = to_tensor(var).reshape(-1, 1)
        # 这里为什么用1-var？
        # self.node_attr_mat = 1-var
        # 差距容忍度
        self.D = 1
        self.the_nodes = None
    # 计算转移概率
    # weights是当前问题到各邻居的权重，ngbrs是当前节点的邻居，nums是当前节点的邻居数，the_weights是上一步节点的权重值，flag是1表示K-Q，0表示Q-K
    def get_prob(self, weights, ngbrs, nums, the_weights,flag):
        if the_weights is not None:
            # 难度之差越小,则概率越大
            if flag ==1:  #K-Q，公式(8)
                # 计算当前问题到各邻居的难度之差，这是考虑边权重的系数
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights-the_weights), to_tensor([self.D]))
                # prob = torch.ones(weights.size()).to(device)
            else:  #Q-K
                prob = to_tensor([1]) - torch.true_divide(torch.abs(weights - the_weights), to_tensor([self.D]))
                # prob = torch.ones(weights.size()).to(device)
                # prob2一个 ngbrs形状的，里面的值为每个ngbrs的索引对应的self.node_attr_mat中的细粒度
                # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
                # print(prob2,prob2.shape)
                # 公式(10) 综合权重
                # prob *= prob2
        else:
            # 如果没有权重，则默认边权重为1
            prob = torch.ones(weights.size()).to(device)
            # prob2 = F.embedding(ngbrs, self.node_attr_mat).squeeze(2)
            # prob*=prob2

        # 防止回溯，即防止重复采样同一节点
        # self.the_nodes是上一个节点的索引，防止回溯造成取回路震荡
        if self.the_nodes is not None:
            # 标记那些节点是上一个节点
            mask1 = (ngbrs == self.the_nodes)
            # 随机数大于1/nums的，才会被采样
            mask2 = torch.rand(nums.size()).to(device) > torch.true_divide(to_tensor([1]), nums)
            # 可以跳回上一个节点但概率极低，取值为-1e32
            prob = prob.masked_fill(mask1 & mask2, -1e32)

        # 令填充的部分采样概率为0
        # 生成一个从0到max_ngbr_num-1的二维张量，形状为[1, max_ngbr_num]  二维是由于unsqueeze(0)
        x = torch.unsqueeze(torch.arange(0, self.max_ngbr_num).to(device), 0) # [[ 0,1,2,3]]
        # x与nums进行比较，x中为索引，nums为每个节点的邻居数，大于邻居数的索引，对应的prob取-1e32 也就是为填充的部分，添加概率为0
        mask = x >= nums

        prob = prob.masked_fill(mask, -1e32)
        # 对各个节点的概率进行归一化，得到最终的转移概率
        prob = F.softmax(prob, dim=1)
        print(prob,prob.shape)
        return prob  # [v_num * walk_num, max_ngbr_num]

    # flag 1是K-Q，0是Q-K the_weights是上一步节点的权重值 v是节点索引列表
    def get_next(self, v, the_weights,flag):
        # v是[0,0,0,01,1,1,1]
        # 根据节点得到 对应的权重、邻居数、邻居
        # 节点v到各个邻居的权重
        expand_pad_weights = self.weights[v]
        # 节点v的邻居数
        expand_ngbr_nums = self.ngbr_nums[v]
        # 节点v的邻居
        expand_pad_ngbrs = self.ngbrs[v]

        # print('expand_pad_weights:',expand_pad_weights)
        # print('expand_ngbr_nums:',expand_ngbr_nums)
        # print('expand_pad_ngbrs:',expand_pad_ngbrs)
        #  返回每个邻居之间的转移概率
        expand_pad_prob = self.get_prob(expand_pad_weights, expand_pad_ngbrs, expand_ngbr_nums, the_weights,flag)
        # print('expand_pad_prob：',expand_pad_prob ) 按多项分布采样，每个样本（节点）返回下一跳的索引
        next_index = torch.multinomial(expand_pad_prob, num_samples=1)  # 按概率采样每行的索引
        # print('next_index',next_index) 根据next_index索引，从expand_pad_ngbrs中取出对应的节点
        self.the_nodes = torch.gather(expand_pad_ngbrs, 1, next_index) # 根据每行的索引对应到节点
        # print('nodes:',self.the_nodes) 排成一排
        next_v = self.the_nodes.flatten() # 平铺为新的一排 k1,k1,k1
        # print('next_v',next_v) 同上 取出节点v到下一跳节点的权重
        the_weights = torch.gather(expand_pad_weights, 1, next_index) # 根据采样索引读取的对应的题目难度
        # print('the_weights:',the_weights) 
        # 返回下一跳节点的索引和权重
        return next_v, the_weights

class QKQ_Walker:
    def __init__(self):
        # self.D = 1  # D notates the biggest distance between diff
        self.qk_edge_list = []
        self.kq_edge_list = []
        self.num_ques = None
        self.num_skill = None

        self.read_data()

    def read_data(self):
        # 读图 ques-skill.csv
        ques_skill_df = pd.read_csv(os.path.join(data_path, data_set, "graph", "ques_skill.csv"))
        # 读权重信息 得到题目难度，将字典转换为python类型
        with open(os.path.join(data_path, data_set, "attribute", "quesID2diffValue_dict.txt")) as f:
            ques2diff_dict = eval(f.read())
            
        # 问题节点的数目是ques_skill_df中ques不重复的数目 skill节点的数目是ques_skill_df中skill不重复的数目 不过这里是集合类型
        # 所以数目都是唯一的
        self.num_ques = len(set(ques_skill_df['ques']))
        self.num_skill = len(set(ques_skill_df['skill']))

        for index, row in ques_skill_df.iterrows():
            # 获取quesID skillID 以及对应的难度 并将 str类型转换为int类型
            quesID, skillID = int(row['ques']), int(row['skill'])
            diffValue = ques2diff_dict[quesID]
            # 构建q-k，k-q边及边上的权重，三元组形式
            self.qk_edge_list.append((quesID, skillID, diffValue))
            self.kq_edge_list.append((skillID, quesID, diffValue))

    # 构建游走路径
    def create_paths(self, walk_num=10, walk_len=80):
        # 构建QK_Walker和KQ_walker
        QK_Walker = WeightedWalker(v_num=self.num_ques, edges=self.qk_edge_list)
        KQ_Walker = WeightedWalker(v_num=self.num_skill, edges=self.kq_edge_list)
        # 0-10 来10行在转置降维 一行num_ques*walk_num列
        # 生成所有问题节点的索引，每个起始问题节点游走walk_num次，排成一排
        next_q = torch.arange(self.num_ques).to(device).repeat(walk_num, 1).T.flatten()
        print(('next_q:',next_q))
        
        paths = [next_q]
        # 将每个起始节点索引行中的每个元素，分成单独的一个[],
        KQ_Walker.the_nodes = next_q.unsqueeze(1)
        the_weights = None
        for i in range(1, walk_len):
            print("%dth hop" % i)
            next_k, the_weights = QK_Walker.get_next(next_q, the_weights,flag=0)
            next_q, the_weights = KQ_Walker.get_next(next_k, the_weights,flag=1)
            paths.append(next_q)
        #维度对其和拼接
        paths = [path.unsqueeze(-1) for path in paths]
        paths = torch.cat(paths, dim=-1)
        paths = paths.view(-1, walk_len)
        # 返回得到的只有问题节点
        return paths


if __name__ == "__main__":
    data_path = "E:/Study/SimKT/SimKT/data"
    for data_set in ["ASSIST09"]:
        save_folder = "../pre_emb/%s/walks" % data_set
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        D_dict = {'qk': 1}
        # decay_rate = 0.5
        walker = QKQ_Walker()
        for num_walks in [10]:
            for walk_length in [80]:
                t = time.time()
                save_file = "walks_qkq_contDiff_pos_%d_%d.txt" % (num_walks, walk_length)
                with open(os.path.join(save_folder, save_file), 'w') as f:
                    paths1 = walker.create_paths(num_walks, walk_length) # 游走路径
                    # 将游走的节点列表转换为python列表 写进文件中去 形成字符串，这个过程是在cpu上运行的
                    for path1 in paths1.cpu().detach().tolist():
                        f.write(','.join([str(e) for e in path1]) + '\n')
                print("time consuming: %d seconds" % (time.time() - t))
