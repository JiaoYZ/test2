import numpy as np
import pandas as pd
import time
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# 自动选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def to_tensor(data):
    """辅助函数：将数据转为 Tensor 并移动到设备"""
    # 如果已经是 Tensor，直接移动；如果是列表/数组，先转换
    if torch.is_tensor(data):
        return data.to(device)
    return torch.tensor(data).to(device)

class WeightedWalker:
    """
    通用加权随机游走器 (CUDA版)
    融合了 SimKT 的属性加权采样思想与工程健壮性
    """
    def __init__(self, v_num, edges, node_attr=None):
        self.v_num = v_num
        self.node_attr = node_attr  # [v_num, 1] 的 Tensor，存储节点属性（如能力值）
        
        # 1. 构建邻接表
        ngbrs = [[] for _ in range(v_num)]
        weights = [[] for _ in range(v_num)]
        
        for u, v, w in edges:
            u, v = int(u), int(v)
            if u < v_num: # 安全检查，防止ID越界
                ngbrs[u].append(v)
                weights[u].append(w)
        
        # 2. 健壮性处理：孤立节点添加自环 (来自您的代码逻辑)
        # 防止某个节点没有邻居导致采样报错
        for i in range(v_num):
            if len(ngbrs[i]) == 0:
                ngbrs[i].append(i) 
                weights[i].append(1.0) # 默认权重
        
        # 3. 转换为 Tensor 并 Padding
        # batch_first=True -> shape: [v_num, max_degree]
        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True, padding_value=0)
        self.weights = pad_sequence([to_tensor(s) for s in weights], batch_first=True, padding_value=0)
        
        # 记录真实邻居数量，用于 mask
        self.ngbr_nums = to_tensor([len(n) for n in ngbrs]).unsqueeze(1)
        self.max_ngbr_num = self.ngbrs.shape[1]
        
        self.the_nodes = None

    def get_prob(self, weights, nums, ngbrs_indices):
        """
        计算采样概率
        SimKT 核心：Probability ~ EdgeWeight * NodeAttribute
        """
        # --- A. 基础概率 ---
        # 如果边权重代表差异(diff)，则用 1-abs(w)；如果代表强度，直接用 w
        # 这里为了通用性，暂设为均匀分布 (或基于 weights)
        # SimKT 中 Q-U 边通常是同质的，所以基础概率是 1
        prob = torch.ones(weights.size()).to(device)
        
        # --- B. 注入灵魂：结合节点属性 ---
        if self.node_attr is not None:
            # 根据邻居索引，查找邻居的属性值
            # ngbrs_indices: [batch, max_degree] -> attr_vals: [batch, max_degree, 1]
            attr_vals = F.embedding(ngbrs_indices, self.node_attr).squeeze(-1)
            
            # 概率 = 基础概率 * 节点属性值
            # 例如：Q->U 时，能力值匹配度(或能力值本身)越高，被采样概率越大
            prob = prob * attr_vals
            
        # --- C. Mask 掉 Padding 部分 ---
        # 生成 mask: [1, max_degree] >= [batch, 1]
        x = torch.arange(self.max_ngbr_num).unsqueeze(0).to(device)
        mask_pad = x >= nums
        prob = prob.masked_fill(mask_pad, 0)
        
        # --- D. 归一化 ---
        row_sum = torch.sum(prob, dim=1, keepdim=True)
        
        # 健壮性：防止 row_sum 为 0 (例如属性全为0，或者被mask完了)
        # 如果全为0，则退化为均匀分布
        mask_zero = (row_sum == 0)
        if mask_zero.any():
            prob = prob.masked_fill(mask_zero, 1.0)
            prob = prob.masked_fill(mask_pad, 0) # 再次 mask padding
            row_sum = torch.sum(prob, dim=1, keepdim=True)
            
        prob = torch.true_divide(prob, row_sum)
        return prob

    def get_next(self, v):
        """
        执行一步游走
        v: 当前节点索引 [batch_size]
        """
        expand_pad_ngbrs = self.ngbrs[v]     
        expand_ngbr_nums = self.ngbr_nums[v] 
        expand_weights = self.weights[v]    
        
        # 计算概率 (传入邻居索引以获取属性)
        expand_pad_prob = self.get_prob(expand_weights, expand_ngbr_nums, expand_pad_ngbrs)
        
        # 采样
        next_index = torch.multinomial(expand_pad_prob, num_samples=1)
        
        # 获取下一个节点
        self.the_nodes = torch.gather(expand_pad_ngbrs, 1, next_index)
        next_v = self.the_nodes.squeeze(1)
        return next_v

class QUCUQ_Walker:
    def __init__(self, data_path, data_set, targe_cluster_num):
        self.data_path = data_path
        self.data_set = data_set
        self.targe_cluster_num = targe_cluster_num

        # 边列表容器
        self.qu_edge_list_correct = [] 
        self.qu_edge_list_wrong = []   
        self.uq_edge_list_correct = []
        self.uq_edge_list_wrong = []
        self.uc_edge_list = []
        self.cu_edge_list = []
        
        self.num_stu = 0
        self.num_ques = 0
        self.num_cluster = 0
        
        # 节点属性容器 (SimKT Soul)
        self.stu_attr = None   # 用于 Q->U 采样
        self.ques_attr = None  # 用于 U->Q 采样

        self.read_data()
        self.load_attributes() # 加载连续值属性
    
    def load_attributes(self):
        """加载连续值属性文件并归一化"""
        print("Loading node attributes (SimKT continuous values)...")
        
        # 1. 加载学生能力 (stu_abi_w.csv)
        path_stu_abi = os.path.join(self.data_path, self.data_set, "graph", "stu_abi_w.csv")
        if os.path.exists(path_stu_abi):
            df = pd.read_csv(path_stu_abi)
            # 假设第2列是能力值
            vals = df.iloc[:, 1].values 
            vals_tensor = torch.tensor(vals, dtype=torch.float32).to(device)
            
            # SimKT 逻辑：计算能力值与其均值的距离，经过 Sigmoid 归一化
            # 这样能力越接近“典型值”或越高，概率越大（视具体逻辑，这里参考 QUQ_Walker_wgt）
            avg = torch.mean(vals_tensor)
            # 这里使用 sigmoid(-abs(x - avg)) 表示越接近均值权重越高(密集区)，或者直接用 sigmoid(x)
            self.stu_attr = torch.sigmoid(-torch.abs(vals_tensor - avg)).unsqueeze(1)
            
            # 健壮性：补齐维度以防 ID 越界
            if self.stu_attr.shape[0] < self.num_stu:
                padding = torch.zeros(self.num_stu - self.stu_attr.shape[0], 1).to(device)
                self.stu_attr = torch.cat([self.stu_attr, padding], dim=0)
            print(f"Loaded Student Attributes: {self.stu_attr.shape}")
        else:
            print("Warning: stu_abi_w.csv not found, using uniform sampling for Students.")
        
        # 2. 加载题目区分度 (ques_discvalue.csv)
        path_ques_disc = os.path.join(self.data_path, self.data_set, "graph", "ques_discvalue.csv")
        if os.path.exists(path_ques_disc):
            df = pd.read_csv(path_ques_disc)
            vals = df.iloc[:, 1].values
            vals_tensor = torch.tensor(vals, dtype=torch.float32).to(device)
            
            # SimKT 逻辑：区分度越高，被访问概率越大
            self.ques_attr = torch.sigmoid(vals_tensor).unsqueeze(1)
            
            if self.ques_attr.shape[0] < self.num_ques:
                padding = torch.zeros(self.num_ques - self.ques_attr.shape[0], 1).to(device)
                self.ques_attr = torch.cat([self.ques_attr, padding], dim=0)
            print(f"Loaded Question Attributes: {self.ques_attr.shape}")
        else:
            print("Warning: ques_discvalue.csv not found, using uniform sampling for Questions.")

    def read_data(self):
        print(f"Reading Graph Data for Cluster {self.targe_cluster_num}...")
        path_stu_ques = os.path.join(self.data_path, self.data_set, "graph", "stu_ques.csv")
        path_stu_cluster = os.path.join(self.data_path, self.data_set, "graph", "stu_cluster_%d.csv" % self.targe_cluster_num)

        df_stu_ques = pd.read_csv(path_stu_ques)
        df_stu_cluster = pd.read_csv(path_stu_cluster)

        # 取交集
        stu_set = set(df_stu_ques['stu']) & set(df_stu_cluster['stu'])
        df_stu_ques = df_stu_ques[df_stu_ques['stu'].isin(stu_set)]
        df_stu_cluster = df_stu_cluster[df_stu_cluster['stu'].isin(stu_set)]

        # 确定节点数量 (健壮性：取 max+1)
        self.num_stu = max(df_stu_ques['stu'].max(), df_stu_cluster['stu'].max()) + 1
        self.num_ques = df_stu_ques['ques'].max() + 1
        self.num_cluster = df_stu_cluster['cluster'].max() + 1
        
        print(f"Nodes: Stu={self.num_stu}, Ques={self.num_ques}, Cluster={self.num_cluster}")
        
        # 构建边列表
        for index, row in df_stu_ques.iterrows():
            stuID, quesID = int(row['stu']), int(row['ques'])
            answerState = int(row['correct']) 
            
            if answerState == 1:
                self.qu_edge_list_correct.append((quesID, stuID, 1))
                self.uq_edge_list_correct.append((stuID, quesID, 1))
            else:
                self.qu_edge_list_wrong.append((quesID, stuID, 1))
                self.uq_edge_list_wrong.append((stuID, quesID, 1))
        
        for index, row in df_stu_cluster.iterrows():
            stuID, clusterID = int(row['stu']), int(row['cluster'])
            self.uc_edge_list.append((stuID, clusterID, 1))
            self.cu_edge_list.append((clusterID, stuID, 1))

    def create_paths(self, walk_num=5, walk_len=80):
        print("Initializing Walkers with attributes...")
        
        # 1. Correct Walker
        # Q->U: 使用 stu_attr 加权
        QU_Walker_Cor = WeightedWalker(v_num=self.num_ques, edges=self.qu_edge_list_correct, node_attr=self.stu_attr)
        # U->Q: 使用 ques_attr 加权
        UQ_Walker_Cor = WeightedWalker(v_num=self.num_stu, edges=self.uq_edge_list_correct, node_attr=self.ques_attr)
        
        # 2. Wrong Walker
        QU_Walker_Wro = WeightedWalker(v_num=self.num_ques, edges=self.qu_edge_list_wrong, node_attr=self.stu_attr)
        UQ_Walker_Wro = WeightedWalker(v_num=self.num_stu, edges=self.uq_edge_list_wrong, node_attr=self.ques_attr)
        
        # 3. Common Walker
        # U->C: 均匀或基于边权重
        UC_Walker = WeightedWalker(v_num=self.num_stu, edges=self.uc_edge_list, node_attr=None)
        # C->U: 可以均匀，也可以优先选能力匹配的学生 (这里为了多样性暂不加权，或者你可以传入 stu_attr)
        CU_Walker = WeightedWalker(v_num=self.num_cluster, edges=self.cu_edge_list, node_attr=None)

        print("Starting Random Walks...")
        start_nodes = torch.arange(self.num_ques).to(device).repeat(walk_num, 1).T.flatten()
        
        # 生成正确路径
        paths_cor = self._run_walk_logic(
            start_nodes, walk_len, 
            QU_Walker_Cor, UC_Walker, CU_Walker, UQ_Walker_Cor
        )
        
        # 生成错误路径
        paths_wro = self._run_walk_logic(
            start_nodes, walk_len, 
            QU_Walker_Wro, UC_Walker, CU_Walker, UQ_Walker_Wro
        )
        
        # 恢复维度 [num_ques, walk_num, walk_len]
        paths_cor = paths_cor.view(self.num_ques, walk_num, walk_len)
        paths_wro = paths_wro.view(self.num_ques, walk_num, walk_len)

        # 拼接结果 [num_ques, 2 * walk_num, walk_len]
        combined_paths = torch.cat([paths_cor, paths_wro], dim=1)
        final_paths = combined_paths.view(-1, walk_len)
        return final_paths

    def _run_walk_logic(self, start_nodes, walk_len, QU, UC, CU, UQ):
        next_q = start_nodes
        paths = [next_q]
        
        # Q -> U -> C -> U -> Q 循环
        # 注意：这里 i 控制的是生成的 Sequence 长度 (即保存多少个Q)
        # 每次循环完成一整套 Q-U-C-U-Q 跳转
        for i in range(1, walk_len):
            next_u = QU.get_next(next_q)
            next_c = UC.get_next(next_u)
            next_u2 = CU.get_next(next_c)
            next_q = UQ.get_next(next_u2)
            paths.append(next_q)

        # 堆叠路径
        paths = [path.unsqueeze(-1) for path in paths]
        paths = torch.cat(paths, dim=-1) # [total_walks, walk_len]
        return paths

if __name__ == "__main__":
    # 配置
    # 请根据您的实际路径修改
    data_path_root = "E:/Study/SimKT/SimKT/data" 
    dataset_name = "EdNet" 
    cluster_n = 120
    
    n_walks = 10 # 每个题目生成多少条路径 (正误各10条，共20条)
    len_walk = 80 # 路径长度 (节点数)

    save_folder = f"../pre_emb/{dataset_name}/walks"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    t_start = time.time()
    
    # 初始化并运行
    walker = QUCUQ_Walker(data_path_root, dataset_name, cluster_n)
    all_paths = walker.create_paths(walk_num=n_walks, walk_len=len_walk)
    
    # 保存文件
    # 文件名格式参考: walks_qucuq_cw_{num_cluster}_{num_walks}_{walk_len}.txt
    out_file_name = os.path.join(save_folder, f'walks_qucuq_cw_{cluster_n}_{n_walks}_{len_walk}.txt')
    
    print(f"Saving {all_paths.shape[0]} paths to {out_file_name}...")
    
    with open(out_file_name, 'w') as f:
        # 转回 CPU 列表写入
        path_list = all_paths.cpu().numpy().tolist()
        for p in path_list:
            f.write(','.join([str(n) for n in p]) + '\n')
            
    print(f"Total time consuming: {time.time() - t_start:.2f}s")