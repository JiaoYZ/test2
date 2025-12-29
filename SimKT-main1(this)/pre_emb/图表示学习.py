import torch
import numpy as np
import os 
current_dir = os.path.dirname(__file__)
emb_path = os.path.join(current_dir, 'ASSIST09/emb/qkckq_contDiff_64_10_80_128_3.emb.npy')
device = torch.device('cuda:0')
mp_npy = np.load(emb_path)
# mp_npy = np.load('D:\Study\Study_Paper\code\SimKT-main\pre_emb\ASSIST09\emb\qkckq_contDiff_64_10_80_128_3.emb.npy')
mp_dict=dict()
mp_dict['quq']= torch.from_numpy(mp_npy).to(device)

if __name__ == "__main__":
    # 测试代码
    print("嵌入矩阵形状:", mp_dict['quq'].shape)  # 应输出 (题目数, 128)
    print("设备信息:", mp_dict['quq'].device)    # 应显示cuda:0
    print("示例数据:", mp_dict['quq'][:3,:5])    # 查看前3题的前5维嵌入

# if __name__ == "__main__":
#     # 测试代码
#     print("嵌入矩阵形状:", mp_dict['quq'].shape)  # 应输出 (题目数, 128)
#     print("设备信息:", mp_dict['quq'].device)    # 应显示cuda:0
#     print("示例数据:", mp_dict['quq'][:3,:5])    # 查看前3题的前5维嵌入
    
#     # 添加嵌入差异性分析代码
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_score
    
#     # 将嵌入转移到CPU并转换为numpy数组以便分析
#     embeddings = mp_dict['quq'].cpu().numpy()
    
#     # 使用PCA降维到2D进行可视化
#     pca = PCA(n_components=2)
#     embeddings_2d = pca.fit_transform(embeddings)
    
#     # 绘制散点图
#     plt.figure(figsize=(10, 8))
#     plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
#     plt.title("PCA Visualization of Question Embeddings")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.grid(True)
#     plt.savefig("embeddings_pca.png")
#     print("PCA可视化已保存为embeddings_pca.png")
    
#     # 聚类分析示例（假设聚类数为10）
#     k = 58
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(embeddings)
    
#     # 计算轮廓系数
#     silhouette_avg = silhouette_score(embeddings, cluster_labels)
#     print(f"平均轮廓系数: {silhouette_avg}")
